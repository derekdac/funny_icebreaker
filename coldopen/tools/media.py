"""Authentic specimen image retrieval via Wikipedia/Wikimedia pageimages API."""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from coldopen.memory.store import (
    ColdOpenStore,
    download_draft_manifest_from_gcs,
    get_export_bucket,
    upload_draft_image_to_gcs,
    upload_draft_manifest_to_gcs,
)
from coldopen.observability.logging import get_logger, log_tool_intent, log_tool_outcome
from coldopen.observability.tracing import traced_span
from coldopen.tools.schemas import SpecimenResult, ToolError

_LOGGER = get_logger("coldopen.tools.media")
_USER_AGENT = "ColdOpenBot/0.1 (https://github.com/google/coldopen; open-source icebreaker deck builder)"
_RETRY_STATUS_CODES = frozenset({429, 503})
_MAX_RETRIES = 3
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB cap to prevent OOM on full-res Commons originals


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()[:48] or "specimen"


def _get_genai_client() -> Any | None:
    """Return a google-genai Client for API-key or Vertex AI ADC environments."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        return genai.Client(api_key=api_key)
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if project:
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
        return genai.Client(vertexai=True, project=project, location=location)
    return None


def _retry_delay_seconds(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return min(max(float(retry_after), 0.1), 5.0)
        except ValueError:
            pass
    return min(0.5 * (2**attempt), 4.0)


async def _async_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = _MAX_RETRIES,
) -> httpx.Response:
    """Perform an async HTTP GET with exponential backoff on 429/503."""
    last_resp: httpx.Response | None = None
    for attempt in range(max_retries):
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code not in _RETRY_STATUS_CODES:
            resp.raise_for_status()
            return resp
        last_resp = resp
        if attempt < max_retries - 1:
            await asyncio.sleep(_retry_delay_seconds(resp, attempt))
    assert last_resp is not None
    last_resp.raise_for_status()
    return last_resp


def _sync_get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = _MAX_RETRIES,
) -> httpx.Response:
    """Perform a sync HTTP GET with exponential backoff on 429/503."""
    last_resp: httpx.Response | None = None
    for attempt in range(max_retries):
        resp = client.get(url, params=params, headers=headers)
        if resp.status_code not in _RETRY_STATUS_CODES:
            resp.raise_for_status()
            return resp
        last_resp = resp
        if attempt < max_retries - 1:
            time.sleep(_retry_delay_seconds(resp, attempt))
    assert last_resp is not None
    last_resp.raise_for_status()
    return last_resp


async def _async_download_image_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int = _MAX_IMAGE_BYTES,
    max_retries: int = _MAX_RETRIES,
) -> bytes | None:
    """Download binary image bytes with Accept: image/*, 429/503 retry, and byte cap."""
    headers = {"Accept": "image/*"}
    for attempt in range(max_retries):
        async with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code in _RETRY_STATUS_CODES:
                if attempt < max_retries - 1:
                    delay = _retry_delay_seconds(resp, attempt)
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        _LOGGER.warning("Skipping oversized image (%s bytes > %d): %s", content_length, max_bytes, url)
                        return None
                except ValueError:
                    pass
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    _LOGGER.warning("Aborting oversized image download (> %d bytes): %s", max_bytes, url)
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    return None


def _sync_download_image_bytes(
    client: httpx.Client,
    url: str,
    *,
    max_bytes: int = _MAX_IMAGE_BYTES,
    max_retries: int = _MAX_RETRIES,
) -> bytes | None:
    """Synchronously download binary image bytes with Accept: image/*, retry, and byte cap."""
    headers = {"Accept": "image/*"}
    for attempt in range(max_retries):
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code in _RETRY_STATUS_CODES:
                if attempt < max_retries - 1:
                    time.sleep(_retry_delay_seconds(resp, attempt))
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        _LOGGER.warning("Skipping oversized image (%s bytes > %d): %s", content_length, max_bytes, url)
                        return None
                except ValueError:
                    pass
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    _LOGGER.warning("Aborting oversized image download (> %d bytes): %s", max_bytes, url)
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    return None


def _save_as_png(raw_bytes: bytes, dest_path: Path) -> str | None:
    """Validate and normalize image bytes to PNG so python-pptx embeds every image cleanly."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
            png_path = dest_path.with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(png_path, format="PNG")
            return str(png_path)
    except Exception:
        return None


async def _query_wikipedia_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await _async_get_with_retry(
        client,
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
        headers={"Accept": "application/json"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {}).get("source")
        if thumb:
            return str(thumb)
    return None


async def _query_commons_thumb(client: httpx.AsyncClient, query: str) -> str | None:
    resp = await _async_get_with_retry(
        client,
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
        headers={"Accept": "application/json"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        for info in page.get("imageinfo", []):
            mime = str(info.get("mime", ""))
            if mime.startswith("image/") and "svg" not in mime:
                return str(info.get("thumburl") or info.get("url") or "")
    return None


def _verify_image_likeness(image_path: str, term: str, hint: str) -> bool:
    """Use Flash Vision to confirm the image matches the entity and is Safe For Work.

    Skips remote verification only in offline local dev where neither GOOGLE_API_KEY
    nor GOOGLE_CLOUD_PROJECT is configured. Fails closed (returns False) on API or
    parse errors when a GenAI client is configured.
    """
    try:
        client = _get_genai_client()
    except Exception as exc:
        _LOGGER.warning("GenAI client initialization failed during likeness check: %s", exc)
        return False

    if client is None:
        return True  # Skip remote vision check in offline local dev / unit tests

    try:
        from google.genai import types

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
        answer = (resp.text or "").strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        _LOGGER.warning("Vision likeness/SFW check failed closed for '%s': %s", term, exc)
        return False


def _generate_gemini_image(term: str, hint: str, dest_path: Path) -> str | None:
    """Generate a clean studio photo fallback via Gemini/Imagen when Wikimedia has no match."""
    try:
        client = _get_genai_client()
    except Exception as exc:
        _LOGGER.warning("GenAI client initialization failed for Imagen fallback: %s", exc)
        return None

    if client is None:
        return None

    try:
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
    except Exception as exc:
        _LOGGER.warning("Imagen fallback failed for '%s': %s", term, exc)
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
                raw_bytes = await _async_download_image_bytes(client, thumb_url)
                if raw_bytes:
                    saved = await asyncio.to_thread(_save_as_png, raw_bytes, dest_path)
                    if saved and await asyncio.to_thread(_verify_image_likeness, saved, term, hint):
                        return saved
        except httpx.HTTPError:
            pass

    return await asyncio.to_thread(_generate_gemini_image, term, hint, dest_path)


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

        query = f"{term.strip()} {disambiguation_hint.strip()}".strip()
        try:
            with httpx.Client(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                search_resp = _sync_get_with_retry(
                    client,
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
                    headers={"Accept": "application/json"},
                )
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

                raw_bytes = _sync_download_image_bytes(client, str(thumb_url))
                if not raw_bytes:
                    res = SpecimenResult(term=term, found=False)
                    log_tool_outcome(
                        _LOGGER,
                        trace_id=trace_id,
                        agent_name="asset_likeness_agent",
                        tool_name="fetch_wikimedia_specimen",
                        status="ok",
                        outcome_summary="image_skipped_or_oversized",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                    return res.model_dump()

                dest_dir = Path(output_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / _slugify(term)
                saved = _save_as_png(raw_bytes, dest_path)
                if not saved or not _verify_image_likeness(saved, term, disambiguation_hint):
                    res = SpecimenResult(term=term, found=False)
                    log_tool_outcome(
                        _LOGGER,
                        trace_id=trace_id,
                        agent_name="asset_likeness_agent",
                        tool_name="fetch_wikimedia_specimen",
                        status="ok",
                        outcome_summary="failed_validation_or_likeness",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                    return res.model_dump()

                res = SpecimenResult(
                    term=term,
                    local_path=saved,
                    source_url=str(thumb_url),
                    found=True,
                )
                log_tool_outcome(
                    _LOGGER,
                    trace_id=trace_id,
                    agent_name="asset_likeness_agent",
                    tool_name="fetch_wikimedia_specimen",
                    status="ok",
                    outcome_summary=f"downloaded={saved}",
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


async def fetch_wikimedia_specimens_batch(
    draft_id: str,
    output_dir: str | None = None,
    db_path: str | None = None,
    max_concurrency: int = 6,
) -> dict[str, object]:
    """Concurrently fetch Wikimedia reveal thumbnails for every item in a saved draft.

    Args:
        draft_id: Identifier of the saved DeckManifest in SQLite (or GCS fallback).
        output_dir: Optional directory where downloaded specimen images are saved.
            Defaults to a draft_id-scoped temporary directory rather than shared assets/.
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
            manifest_dict = await asyncio.to_thread(download_draft_manifest_from_gcs, draft_id)
            if manifest_dict:
                store.save_draft_manifest(draft_id, manifest_dict)

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
        export_bucket = get_export_bucket()
        draft_slug = _slugify(draft_id)

        # Scope downloads to a draft_id-specific directory (never shared assets/).
        # When GCS persistence is active and no explicit output_dir was requested,
        # use an auto-cleaning TemporaryDirectory to avoid container tmpfs leaks.
        ephemeral = bool(export_bucket) and (output_dir is None or output_dir == "assets")
        dir_ctx: Any = (
            tempfile.TemporaryDirectory(prefix=f"coldopen_{draft_slug}_")
            if ephemeral
            else nullcontext(
                str(
                    (Path(output_dir) / draft_slug)
                    if (output_dir is not None and output_dir != "assets")
                    else (Path(tempfile.gettempdir()) / "coldopen_assets" / draft_slug)
                )
            )
        )

        with dir_ctx as work_dir_str:
            dest_dir = Path(work_dir_str)
            dest_dir.mkdir(parents=True, exist_ok=True)
            sem = asyncio.Semaphore(max_concurrency)

            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
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
            for idx, (item, local_path) in enumerate(zip(items, paths)):
                resolved_path = local_path
                if local_path and export_bucket:
                    gcs_uri = await asyncio.to_thread(
                        upload_draft_image_to_gcs,
                        draft_id,
                        idx,
                        local_path,
                        export_bucket,
                    )
                    if gcs_uri:
                        resolved_path = gcs_uri
                    elif ephemeral:
                        # Temp dir will be deleted on exit; do not persist a dangling path.
                        resolved_path = None
                item["image_path"] = resolved_path
                if resolved_path:
                    found += 1

        manifest_dict["items"] = items
        store.save_draft_manifest(draft_id, manifest_dict)
        if export_bucket:
            await asyncio.to_thread(upload_draft_manifest_to_gcs, draft_id, manifest_dict, export_bucket)

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
