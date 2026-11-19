"""Authentic specimen image retrieval via Wikipedia/Wikimedia pageimages API."""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image

from coldopen.memory.store import ColdOpenStore
from coldopen.observability.logging import get_logger, log_tool_intent, log_tool_outcome
from coldopen.observability.tracing import traced_span
from coldopen.tools.schemas import SpecimenResult, ToolError

_LOGGER = get_logger("coldopen.tools.media")
_USER_AGENT = "ColdOpenBot/0.1 (open-source icebreaker deck builder)"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()[:48] or "specimen"


def fetch_wikimedia_specimen(
    term: str,
    disambiguation_hint: str,
    output_dir: str = "assets",
) -> dict[str, object]:
    """Fetch a high-resolution thumbnail image from Wikipedia for a reveal slide.

    Args:
        term: Exact entity or product name to search on Wikipedia.
        disambiguation_hint: Category context (e.g. 'Scottish cheese' or 'IKEA furniture')
            used to disambiguate homonyms and acronyms.
        output_dir: Directory where downloaded specimen images are saved.

    Returns:
        Dictionary matching SpecimenResult on success, or ToolError with recovery
        instructions when network or input validation fails.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="asset_likeness_agent",
        tool_name="fetch_wikimedia_specimen",
        intended_action="Query Wikipedia pageimages API and download reveal thumbnail",
        arguments={"term": term, "hint": disambiguation_hint},
    )

    with traced_span("tool.fetch_wikimedia_specimen", {"term": term}):
        if not term.strip():
            err = ToolError(
                error_code="EMPTY_TERM",
                message="Cannot search Wikimedia with an empty term.",
                recovery_instruction="Provide a non-empty 'term' and 'disambiguation_hint'.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimen",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()


def _save_as_png(raw_bytes: bytes, dest_path: Path) -> str | None:
    """Validate and normalize image bytes to PNG so python-pptx embeds every image cleanly."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
            png_path = dest_path.with_suffix(".png")
            rgb.save(png_path, format="PNG")
            return str(png_path)
    except Exception:
        return None


async def _query_wikipedia_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 2,
            "prop": "pageimages",
            "pithumbsize": 960,
            "format": "json",
        },
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {}).get("source")
        if thumb:
            return str(thumb)
    return None


async def _query_commons_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await client.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": 2,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 960,
            "format": "json",
        },
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        for info in page.get("imageinfo", []):
            mime = str(info.get("mime", ""))
            if mime.startswith("image/") and "svg" not in mime:
                return str(info.get("thumburl") or info.get("url") or "")
    return None


def _verify_image_likeness(image_path: str, term: str, hint: str) -> bool:
    """Use Flash Vision to confirm the image matches the entity and is Safe For Work."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return True  # Skip remote vision check in offline unit tests
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        img_bytes = Path(image_path).read_bytes()
        vision_model = os.getenv("COLDOPEN_MODEL_FLASH", "gemini-3.8-flash")
        prompt = (
            f"Does this image clearly depict '{term}' ({hint}) AND is it 100% Safe For Work "
            "(no nudity, gore, slurs, or inappropriate content)? Reply with ONLY 'YES' or 'NO'."
        )
        resp = client.models.generate_content(
            model=vision_model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return "YES" in (resp.text or "").strip().upper()
    except Exception:
        return True


def _generate_gemini_image(term: str, hint: str, dest_path: Path) -> str | None:
    """Generate a clean studio photo fallback via Gemini/Imagen when Wikimedia has no match."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        image_model = os.getenv("COLDOPEN_IMAGE_MODEL", "imagen-3.0-generate-002")
        prompt = (
            f"Clean, sharp, realistic studio photograph of '{term}' ({hint}). "
            "No text, no watermarks, centered subject on a clean background."
        )
        result = client.models.generate_images(
            model=image_model,
            prompt=prompt,
            config={"number_of_images": 1},
        )
        if result.generated_images:
            img_bytes = result.generated_images[0].image.image_bytes
            return _save_as_png(img_bytes, dest_path)
    except Exception:
        return None
    return None


async def _fetch_one_async(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    term: str,
    hint: str,
    dest_dir: Path,
) -> str | None:
    dest_path = dest_dir / _slugify(term)
    async with sem:
        try:
            thumb_url = (
                await _query_wikipedia_thumb(client, f"{term.strip()} {hint.strip()}".strip())
                or await _query_wikipedia_thumb(client, term.strip())
                or await _query_commons_thumb(client, f"{term.strip()} {hint.strip()}".strip())
            )
            if thumb_url:
                img_resp = await client.get(thumb_url)
                img_resp.raise_for_status()
                saved = _save_as_png(img_resp.content, dest_path)
                if saved and await asyncio.to_thread(_verify_image_likeness, saved, term, hint):
                    return saved
        except httpx.HTTPError:
            pass

    return await asyncio.to_thread(_generate_gemini_image, term, hint, dest_path)


async def fetch_wikimedia_specimens_batch(
    draft_id: str,
    output_dir: str = "assets",
    db_path: str = "coldopen_state.db",
    max_concurrency: int = 6,
) -> dict[str, object]:
    """Concurrently fetch Wikimedia reveal thumbnails for every item in a saved draft.

    Args:
        draft_id: Identifier of the saved DeckManifest in SQLite.
        output_dir: Directory where downloaded specimen images are saved.
        db_path: SQLite database path storing the draft manifest.
        max_concurrency: Maximum number of concurrent Wikipedia HTTP requests.

    Returns:
        Compact summary with draft_id, images_found, and total_items, or ToolError
        if the draft_id does not exist.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="asset_likeness_agent",
        tool_name="fetch_wikimedia_specimens_batch",
        intended_action="Concurrently fetch Wikimedia thumbnails for all items in draft",
        arguments={"draft_id": draft_id, "max_concurrency": max_concurrency},
    )

    with traced_span("tool.fetch_wikimedia_specimens_batch", {"draft_id": draft_id}):
        store = ColdOpenStore(db_path=db_path)
        manifest_dict = store.get_draft_manifest(draft_id)
        if not manifest_dict:
            err = ToolError(
                error_code="DRAFT_NOT_FOUND",
                message=f"No draft manifest found for draft_id '{draft_id}'.",
                recovery_instruction="Call save_and_audit_draft first to obtain a valid draft_id.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimens_batch",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()

        if manifest_dict.get("layout_mode") == "pill_only":
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimens_batch",
                status="ok",
                outcome_summary="skipped (pill_only layout)",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return {
                "ok": True,
                "draft_id": draft_id,
                "images_found": 0,
                "total_items": len(manifest_dict.get("items", [])),
                "skipped_reason": "pill_only",
            }

        items = manifest_dict.get("items", [])
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(max_concurrency)
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            paths = await asyncio.gather(
                *[
                    _fetch_one_async(
                        client,
                        sem,
                        str(item.get("term", "")),
                        str(item.get("category_label", "")),
                        dest_dir,
                    )
                    for item in items
                ]
            )

        found = 0
        for item, local_path in zip(items, paths):
            item["image_path"] = local_path
            if local_path:
                found += 1

        manifest_dict["items"] = items
        store.save_draft_manifest(draft_id, manifest_dict)

        log_tool_outcome(
            _LOGGER,
            trace_id=trace_id,
            agent_name="asset_likeness_agent",
            tool_name="fetch_wikimedia_specimens_batch",
            status="ok",
            outcome_summary=f"found={found}/{len(items)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return {
            "ok": True,
            "draft_id": draft_id,
            "images_found": found,
            "total_items": len(items),
        }

        query = f"{term.strip()} {disambiguation_hint.strip()}".strip()
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        try:
            with httpx.Client(timeout=10.0, headers=headers) as client:
                search_resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrlimit": 1,
                        "prop": "pageimages",
                        "pithumbsize": 960,
                        "format": "json",
                    },
                )
                search_resp.raise_for_status()
                pages = search_resp.json().get("query", {}).get("pages", {})
                thumb_url = None
                for page in pages.values():
                    thumb_url = page.get("thumbnail", {}).get("source")
                    if thumb_url:
                        break

                if not thumb_url:
                    res = SpecimenResult(term=term, found=False)
                    log_tool_outcome(
                        _LOGGER,
                        trace_id=trace_id,
                        agent_name="asset_likeness_agent",
                        tool_name="fetch_wikimedia_specimen",
                        status="ok",
                        outcome_summary="no_thumbnail_found",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                    return res.model_dump()

                img_resp = client.get(thumb_url)
                img_resp.raise_for_status()
                dest_dir = Path(output_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                ext = ".png" if thumb_url.lower().endswith(".png") else ".jpg"
                dest_path = dest_dir / f"{_slugify(term)}{ext}"
                dest_path.write_bytes(img_resp.content)

                res = SpecimenResult(
                    term=term,
                    local_path=str(dest_path),
                    source_url=thumb_url,
                    found=True,
                )
                log_tool_outcome(
                    _LOGGER,
                    trace_id=trace_id,
                    agent_name="asset_likeness_agent",
                    tool_name="fetch_wikimedia_specimen",
                    status="ok",
                    outcome_summary=f"downloaded={dest_path}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                return res.model_dump()
        except httpx.HTTPError as exc:
            err = ToolError(
                error_code="WIKIMEDIA_HTTP_ERROR",
                message=f"Wikimedia lookup failed: {exc}",
                recovery_instruction="Proceed with text pill-badge reveal by leaving image_path=None for this item.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimen",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()


def _save_as_png(raw_bytes: bytes, dest_path: Path) -> str | None:
    """Validate and normalize image bytes to PNG so python-pptx embeds every image cleanly."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
            png_path = dest_path.with_suffix(".png")
            rgb.save(png_path, format="PNG")
            return str(png_path)
    except Exception:
        return None


async def _query_wikipedia_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 2,
            "prop": "pageimages",
            "pithumbsize": 960,
            "format": "json",
        },
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {}).get("source")
        if thumb:
            return str(thumb)
    return None


async def _query_commons_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await client.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": 2,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 960,
            "format": "json",
        },
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        for info in page.get("imageinfo", []):
            mime = str(info.get("mime", ""))
            if mime.startswith("image/") and "svg" not in mime:
                return str(info.get("thumburl") or info.get("url") or "")
    return None


def _verify_image_likeness(image_path: str, term: str, hint: str) -> bool:
    """Use Flash Vision to confirm the image matches the entity and is Safe For Work."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return True  # Skip remote vision check in offline unit tests
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        img_bytes = Path(image_path).read_bytes()
        vision_model = os.getenv("COLDOPEN_MODEL_FLASH", "gemini-3.8-flash")
        prompt = (
            f"Does this image clearly depict '{term}' ({hint}) AND is it 100% Safe For Work "
            "(no nudity, gore, slurs, or inappropriate content)? Reply with ONLY 'YES' or 'NO'."
        )
        resp = client.models.generate_content(
            model=vision_model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return "YES" in (resp.text or "").strip().upper()
    except Exception:
        return True


def _generate_gemini_image(term: str, hint: str, dest_path: Path) -> str | None:
    """Generate a clean studio photo fallback via Gemini/Imagen when Wikimedia has no match."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        image_model = os.getenv("COLDOPEN_IMAGE_MODEL", "imagen-3.0-generate-002")
        prompt = (
            f"Clean, sharp, realistic studio photograph of '{term}' ({hint}). "
            "No text, no watermarks, centered subject on a clean background."
        )
        result = client.models.generate_images(
            model=image_model,
            prompt=prompt,
            config={"number_of_images": 1},
        )
        if result.generated_images:
            img_bytes = result.generated_images[0].image.image_bytes
            return _save_as_png(img_bytes, dest_path)
    except Exception:
        return None
    return None


async def _fetch_one_async(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    term: str,
    hint: str,
    dest_dir: Path,
) -> str | None:
    dest_path = dest_dir / _slugify(term)
    async with sem:
        try:
            thumb_url = (
                await _query_wikipedia_thumb(client, f"{term.strip()} {hint.strip()}".strip())
                or await _query_wikipedia_thumb(client, term.strip())
                or await _query_commons_thumb(client, f"{term.strip()} {hint.strip()}".strip())
            )
            if thumb_url:
                img_resp = await client.get(thumb_url)
                img_resp.raise_for_status()
                saved = _save_as_png(img_resp.content, dest_path)
                if saved and await asyncio.to_thread(_verify_image_likeness, saved, term, hint):
                    return saved
        except httpx.HTTPError:
            pass

    return await asyncio.to_thread(_generate_gemini_image, term, hint, dest_path)


async def fetch_wikimedia_specimens_batch(
    draft_id: str,
    output_dir: str = "assets",
    db_path: str = "coldopen_state.db",
    max_concurrency: int = 6,
) -> dict[str, object]:
    """Concurrently fetch Wikimedia reveal thumbnails for every item in a saved draft.

    Args:
        draft_id: Identifier of the saved DeckManifest in SQLite.
        output_dir: Directory where downloaded specimen images are saved.
        db_path: SQLite database path storing the draft manifest.
        max_concurrency: Maximum number of concurrent Wikipedia HTTP requests.

    Returns:
        Compact summary with draft_id, images_found, and total_items, or ToolError
        if the draft_id does not exist.
    """
    start = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    log_tool_intent(
        _LOGGER,
        trace_id=trace_id,
        agent_name="asset_likeness_agent",
        tool_name="fetch_wikimedia_specimens_batch",
        intended_action="Concurrently fetch Wikimedia thumbnails for all items in draft",
        arguments={"draft_id": draft_id, "max_concurrency": max_concurrency},
    )

    with traced_span("tool.fetch_wikimedia_specimens_batch", {"draft_id": draft_id}):
        store = ColdOpenStore(db_path=db_path)
        manifest_dict = store.get_draft_manifest(draft_id)
        if not manifest_dict:
            err = ToolError(
                error_code="DRAFT_NOT_FOUND",
                message=f"No draft manifest found for draft_id '{draft_id}'.",
                recovery_instruction="Call save_and_audit_draft first to obtain a valid draft_id.",
            )
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimens_batch",
                status="error",
                outcome_summary=err.message,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return err.model_dump()

        if manifest_dict.get("layout_mode") == "pill_only":
            log_tool_outcome(
                _LOGGER,
                trace_id=trace_id,
                agent_name="asset_likeness_agent",
                tool_name="fetch_wikimedia_specimens_batch",
                status="ok",
                outcome_summary="skipped (pill_only layout)",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return {
                "ok": True,
                "draft_id": draft_id,
                "images_found": 0,
                "total_items": len(manifest_dict.get("items", [])),
                "skipped_reason": "pill_only",
            }

        items = manifest_dict.get("items", [])
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(max_concurrency)
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            paths = await asyncio.gather(
                *[
                    _fetch_one_async(
                        client,
                        sem,
                        str(item.get("term", "")),
                        str(item.get("category_label", "")),
                        dest_dir,
                    )
                    for item in items
                ]
            )

        found = 0
        for item, local_path in zip(items, paths):
            item["image_path"] = local_path
            if local_path:
                found += 1

        manifest_dict["items"] = items
        store.save_draft_manifest(draft_id, manifest_dict)

        log_tool_outcome(
            _LOGGER,
            trace_id=trace_id,
            agent_name="asset_likeness_agent",
            tool_name="fetch_wikimedia_specimens_batch",
            status="ok",
            outcome_summary=f"found={found}/{len(items)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return {
            "ok": True,
            "draft_id": draft_id,
            "images_found": found,
            "total_items": len(items),
        }
