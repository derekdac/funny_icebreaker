"""Targeted unit tests for media.py deduplication, Vertex AI ADC Vision/Imagen, SQLite WAL/closing,
temp-dir isolated PPTX compilation, corrupt-image resilience, and cross-instance GCS draft fallback.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from PIL import Image
from pptx import Presentation

from coldopen.memory import store as store_mod
from coldopen.memory.store import ColdOpenStore
from coldopen.tools import compiler as compiler_mod
from coldopen.tools import media as media_mod
from coldopen.tools.schemas import DeckManifest, SlideItem


def _png_bytes(width: int = 64, height: int = 48, color: tuple[int, int, int] = (20, 120, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _sample_manifest(team_id: str = "state-team") -> DeckManifest:
    categories = ["A", "A", "B", "A", "B", "B"]
    items = [
        SlideItem(
            term=f"Specimen {idx + 1}",
            category=cat,  # type: ignore[arg-type]
            category_label="Nordic Chair" if cat == "A" else "Rock Band",
            subtitle="Verified binary round item",
            speaker_notes=f"Item {idx + 1}: {'Chair' if cat == 'A' else 'Band'}",
        )
        for idx, cat in enumerate(categories)
    ]
    return DeckManifest(
        team_id=team_id,
        deck_title="Chair or Band?",
        category_a_name="Nordic Chair",
        category_b_name="Rock Band",
        items=items,
    )


class _InMemoryGCSBlob:
    def __init__(self, bucket_store: dict[str, bytes], name: str, *, allow_sign: bool = True) -> None:
        self._store = bucket_store
        self.name = name
        self.content_disposition: str | None = None
        self.content_type: str | None = None
        self._allow_sign = allow_sign

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, data: str | bytes, content_type: str | None = None) -> None:
        self.content_type = content_type
        self._store[self.name] = data.encode("utf-8") if isinstance(data, str) else data

    def upload_from_filename(self, filename: str, content_type: str | None = None) -> None:
        self.content_type = content_type
        self._store[self.name] = Path(filename).read_bytes()

    def download_as_text(self) -> str:
        return self._store[self.name].decode("utf-8")

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self._store[self.name])

    def generate_signed_url(self, **kwargs: Any) -> str:
        if not self._allow_sign:
            raise AttributeError("Credentials do not support signing")
        return f"https://signed.example.com/{self.name}?v4=1"


class _InMemoryGCSBucket:
    def __init__(self, bucket_store: dict[str, bytes], *, allow_sign: bool = True) -> None:
        self._store = bucket_store
        self._allow_sign = allow_sign
        self.blobs: dict[str, _InMemoryGCSBlob] = {}

    def blob(self, name: str) -> _InMemoryGCSBlob:
        if name not in self.blobs:
            self.blobs[name] = _InMemoryGCSBlob(self._store, name, allow_sign=self._allow_sign)
        return self.blobs[name]


class _InMemoryGCSClient:
    def __init__(self, buckets: dict[str, dict[str, bytes]], *, allow_sign: bool = True) -> None:
        self._buckets = buckets
        self._allow_sign = allow_sign
        self._bucket_objs: dict[str, _InMemoryGCSBucket] = {}

    def bucket(self, name: str) -> _InMemoryGCSBucket:
        if name not in self._buckets:
            self._buckets[name] = {}
        if name not in self._bucket_objs:
            self._bucket_objs[name] = _InMemoryGCSBucket(self._buckets[name], allow_sign=self._allow_sign)
        return self._bucket_objs[name]


# ---------------------------------------------------------------------------
# 1. P1 #15 & Wikimedia HTTP reliability
# ---------------------------------------------------------------------------


def test_media_module_has_no_duplicate_definitions() -> None:
    source = Path(media_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn_names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    expected_unique = [
        "fetch_wikimedia_specimen",
        "fetch_wikimedia_specimens_batch",
        "_save_as_png",
        "_query_wikipedia_thumb",
        "_query_commons_thumb",
        "_verify_image_likeness",
        "_generate_gemini_image",
        "_fetch_one_async",
    ]
    for name in expected_unique:
        assert fn_names.count(name) == 1, f"Expected exactly 1 definition of {name}, found {fn_names.count(name)}"


def test_fetch_wikimedia_specimen_retries_and_uses_proper_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    png_data = _png_bytes()
    seen_requests: list[tuple[str, str | None, str | None]] = []
    api_attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((str(request.url), request.headers.get("User-Agent"), request.headers.get("Accept")))
        if "en.wikipedia.org/w/api.php" in str(request.url):
            api_attempts["count"] += 1
            if api_attempts["count"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0.01"})
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": {
                            "1": {"thumbnail": {"source": "https://upload.wikimedia.org/thumb/specimen.jpg"}}
                        }
                    }
                },
            )
        if "upload.wikimedia.org" in str(request.url):
            return httpx.Response(200, content=png_data, headers={"Content-Type": "image/png"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _mock_client(*args: Any, **kwargs: Any) -> httpx.Client:
        assert kwargs.get("follow_redirects") is True
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(media_mod.httpx, "Client", _mock_client)

    res = media_mod.fetch_wikimedia_specimen("Billy", "IKEA bookcase", output_dir=str(tmp_path))
    assert res["ok"] is True
    assert res["found"] is True
    assert Path(str(res["local_path"])).exists()
    assert api_attempts["count"] == 2

    # Check Wikimedia-compliant User-Agent with contact URL and proper Accept headers
    assert any("http" in (ua or "") for _, ua, _ in seen_requests)
    api_reqs = [acc for url, _, acc in seen_requests if "api.php" in url]
    img_reqs = [acc for url, _, acc in seen_requests if "upload.wikimedia.org" in url]
    assert all(acc == "application/json" for acc in api_reqs)
    assert img_reqs == ["image/*"]


@pytest.mark.asyncio
async def test_async_download_enforces_max_image_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 2048, headers={"Content-Length": "2048"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data = await media_mod._async_download_image_bytes(client, "https://example.com/huge.png", max_bytes=1024)
    assert data is None


# ---------------------------------------------------------------------------
# 2. P0 #5 — Vertex AI ADC client selection, fail-closed Vision, Imagen fallback
# ---------------------------------------------------------------------------


def test_get_genai_client_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    import google.genai as genai_pkg

    monkeypatch.setattr(genai_pkg, "Client", _FakeClient)

    # 1. Offline mode (neither key nor project) -> None
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert media_mod._get_genai_client() is None

    # 2. Vertex AI ADC mode (GOOGLE_CLOUD_PROJECT set, no GOOGLE_API_KEY)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    c_vertex = media_mod._get_genai_client()
    assert c_vertex is not None
    assert calls[-1] == {"vertexai": True, "project": "demo-proj", "location": "global"}

    # 3. API key mode takes precedence when GOOGLE_API_KEY is set
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    c_key = media_mod._get_genai_client()
    assert c_key is not None
    assert calls[-1] == {"api_key": "test-key-123"}


def test_verify_image_likeness_fails_closed_on_api_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img_path = tmp_path / "test.png"
    img_path.write_bytes(_png_bytes())

    class _FailingModels:
        def generate_content(self, **kwargs: Any) -> Any:
            raise RuntimeError("Vertex API 503")

    monkeypatch.setattr(media_mod, "_get_genai_client", lambda: SimpleNamespace(models=_FailingModels()))
    assert media_mod._verify_image_likeness(str(img_path), "Billy", "bookcase") is False

    class _NoModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return SimpleNamespace(text="NO")

    monkeypatch.setattr(media_mod, "_get_genai_client", lambda: SimpleNamespace(models=_NoModels()))
    assert media_mod._verify_image_likeness(str(img_path), "Billy", "bookcase") is False

    class _YesModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return SimpleNamespace(text="YES")

    monkeypatch.setattr(media_mod, "_get_genai_client", lambda: SimpleNamespace(models=_YesModels()))
    assert media_mod._verify_image_likeness(str(img_path), "Billy", "bookcase") is True


def test_generate_gemini_image_uses_vertex_adc_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_data = _png_bytes()

    class _FakeImagenModels:
        def generate_images(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                generated_images=[SimpleNamespace(image=SimpleNamespace(image_bytes=png_data))]
            )

    monkeypatch.setattr(media_mod, "_get_genai_client", lambda: SimpleNamespace(models=_FakeImagenModels()))
    saved = media_mod._generate_gemini_image("Billy", "bookcase", tmp_path / "billy")
    assert saved is not None
    assert Path(saved).exists()


# ---------------------------------------------------------------------------
# 3. P1 #8 — SQLite WAL, busy_timeout, COLDOPEN_DB_PATH, contextlib.closing
# ---------------------------------------------------------------------------


def test_store_wal_busy_timeout_env_default_and_closing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_db = tmp_path / "nested" / "env_state.db"
    monkeypatch.setenv("COLDOPEN_DB_PATH", str(env_db))

    store = ColdOpenStore()
    assert store.db_path == str(env_db)
    assert env_db.exists()

    with store._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == 5000
        captured_conn = conn

    # After exiting _connect(), contextlib.closing must have closed the SQLite connection.
    with pytest.raises(sqlite3.ProgrammingError):
        captured_conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# 4. P0 #2, P0 #6, P1 #8 — Corrupt image resilience, temp-dir isolation, GCS fallback
# ---------------------------------------------------------------------------


def test_corrupt_image_does_not_crash_compilation(tmp_path: Path) -> None:
    bad_img = tmp_path / "corrupt.png"
    bad_img.write_bytes(b"not-a-valid-png-file")

    manifest = _sample_manifest("corrupt-team")
    manifest.items[0].image_path = str(bad_img)

    out_pptx = tmp_path / "deck.pptx"
    res = compiler_mod.compile_pptx_deck(
        manifest,
        output_path=str(out_pptx),
        db_path=str(tmp_path / "corrupt.db"),
    )
    assert res["ok"] is True
    assert out_pptx.exists()
    prs = Presentation(str(out_pptx))
    assert len(prs.slides) == 14


@pytest.mark.asyncio
async def test_cross_instance_gcs_draft_and_image_persistence_with_tempdir_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buckets: dict[str, dict[str, bytes]] = {}
    gcs_client = _InMemoryGCSClient(buckets, allow_sign=True)

    monkeypatch.setenv("COLDOPEN_EXPORT_BUCKET", "coldopen-test-bucket")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(store_mod, "_get_gcs_client", lambda: gcs_client)

    # Also patch google.cloud.storage.Client used by _upload_pptx_to_gcs
    import google.cloud.storage as gcs_storage

    monkeypatch.setattr(gcs_storage, "Client", lambda project=None: gcs_client)

    # Instance A: save_and_audit_draft writes to SQLite A + GCS
    db_a = str(tmp_path / "instance_a.db")
    saved = compiler_mod.save_and_audit_draft(
        _sample_manifest("multi-inst-team"),
        banned_category_terms=["chair", "band"],
        db_path=db_a,
    )
    assert saved["ok"] is True
    draft_id = str(saved["draft_id"])
    assert f"coldopen-drafts/{draft_id}/manifest.json" in buckets["coldopen-test-bucket"]

    # Instance B (fresh empty SQLite B): fetch_wikimedia_specimens_batch falls back to GCS
    db_b = str(tmp_path / "instance_b.db")
    png_data = _png_bytes()

    async def _fake_fetch_one(
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        term: str,
        hint: str,
        dest_dir: Path,
    ) -> str | None:
        assert dest_dir.name != "assets"
        out = dest_dir / f"{media_mod._slugify(term)}.png"
        out.write_bytes(png_data)
        return str(out)

    monkeypatch.setattr(media_mod, "_fetch_one_async", _fake_fetch_one)
    batch_res = await media_mod.fetch_wikimedia_specimens_batch(draft_id=draft_id, db_path=db_b)
    assert batch_res["ok"] is True
    assert batch_res["images_found"] == 6
    assert f"coldopen-drafts/{draft_id}/images/0.png" in buckets["coldopen-test-bucket"]

    updated_manifest = json.loads(buckets["coldopen-test-bucket"][f"coldopen-drafts/{draft_id}/manifest.json"])
    assert updated_manifest["items"][0]["image_path"] == f"gs://coldopen-test-bucket/coldopen-drafts/{draft_id}/images/0.png"

    # Instance C (fresh empty SQLite C): compile_and_save_pptx_artifact falls back to GCS,
    # downloads gs:// images into an isolated temp dir, compiles inside temp dir, and cleans up.
    db_c = str(tmp_path / "instance_c.db")
    cwd_dir = tmp_path / "cwd_clean"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    captured_artifacts: list[tuple[str, Any]] = []

    class _FakeToolContext:
        async def save_artifact(self, *, filename: str, artifact: Any) -> int:
            captured_artifacts.append((filename, artifact))
            return 1

    compile_res = await compiler_mod.compile_and_save_pptx_artifact(
        draft_id=draft_id,
        tool_context=_FakeToolContext(),  # type: ignore[arg-type]
        db_path=db_c,
    )
    assert compile_res["ok"] is True
    assert compile_res["artifact_filename"] == "chair_or_band.pptx"
    assert compile_res["artifact_version"] == 1
    assert compile_res["download_url"].startswith("https://signed.example.com/coldopen-decks/")

    # Temp dir must be cleaned up and cwd must remain completely clean!
    assert not Path(str(compile_res["output_path"])).exists()
    assert list(cwd_dir.iterdir()) == []

    # Verify saved artifact is a valid PPTX containing the downloaded picture on reveal slide
    assert len(captured_artifacts) == 1
    pptx_bytes = captured_artifacts[0][1].inline_data.data
    prs = Presentation(io.BytesIO(pptx_bytes))
    assert any(shape.shape_type == 13 for shape in prs.slides[2].shapes)  # 13 = PICTURE


def test_upload_pptx_to_gcs_falls_back_when_signing_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buckets: dict[str, dict[str, bytes]] = {}
    gcs_client = _InMemoryGCSClient(buckets, allow_sign=False)
    import google.cloud.storage as gcs_storage

    monkeypatch.setattr(gcs_storage, "Client", lambda project=None: gcs_client)

    pptx_file = tmp_path / "sample_deck.pptx"
    pptx_file.write_bytes(b"PK\x03\x04fake-pptx")

    url = compiler_mod._upload_pptx_to_gcs(pptx_file, "my-bucket", "draft-abc123")
    assert url == "https://storage.cloud.google.com/my-bucket/coldopen-decks/draft-abc123/sample_deck.pptx"
    blob = gcs_client.bucket("my-bucket").blobs["coldopen-decks/draft-abc123/sample_deck.pptx"]
    assert blob.content_disposition == 'attachment; filename="sample_deck.pptx"'
