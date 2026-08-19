"""Unit tests for repository indexer service."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.indexing.indexer import RepositoryIndexer


def test_extract_chunks_from_file():
    """Test extracting chunks from a sample Python file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        py_file = root / "service.py"
        py_file.write_text(
            "def process_data(items):\n    return [x * 2 for x in items]\n",
            encoding="utf-8",
        )

        indexer = RepositoryIndexer(repo_path=root, repo_id=1)
        chunks = indexer.extract_chunks_from_file(py_file)

        assert len(chunks) >= 1
        assert chunks[0].repository_id == 1
        assert chunks[0].file_path == "service.py"
        assert chunks[0].symbol_name == "process_data"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2


def test_scan_all_chunks():
    """Test scanning an entire multi-file repository structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "module_a.py").write_text("def a(): pass\n", encoding="utf-8")
        (root / "module_b.py").write_text("def b(): pass\n", encoding="utf-8")

        indexer = RepositoryIndexer(repo_path=root, repo_id=2)
        all_chunks = indexer.scan_all_chunks()

        assert len(all_chunks) >= 2
        file_paths = {c.file_path for c in all_chunks}
        assert "module_a.py" in file_paths
        assert "module_b.py" in file_paths


class FakeResult:
    """Minimal stand-in for SQLAlchemy result objects."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    """Records indexer DB interactions without a real database."""

    def __init__(self, existing_rows=None):
        self.existing_rows = existing_rows or []
        self.added = []
        self.committed = False

    async def execute(self, stmt):
        if getattr(stmt, "is_select", False):
            return FakeResult(self.existing_rows)
        return FakeResult([])

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


def _make_repo(tmpdir: str, files: dict, repo_id: int = 3) -> RepositoryIndexer:
    root = Path(tmpdir)
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    return RepositoryIndexer(repo_path=root, repo_id=repo_id)


def _good_provider(monkeypatch):
    """Patch the indexer's embedding provider to return correct 3072-dim vectors."""

    class GoodProvider:
        dimension = 3072

        async def embed_batch(self, texts, batch_size=50):
            return [[0.01] * 3072 for _ in texts]

    monkeypatch.setattr(
        "app.services.indexing.indexer.GeminiEmbeddingProvider",
        lambda *a, **k: GoodProvider(),
    )


@pytest.mark.asyncio
async def test_index_repository_raises_on_incomplete_embeddings(monkeypatch):
    """Indexing must fail instead of inserting chunks when the provider drops vectors."""

    class IncompleteProvider:
        dimension = 3072

        async def embed_batch(self, texts, batch_size=50):
            return [[0.01] * 3072]

    monkeypatch.setattr(
        "app.services.indexing.indexer.GeminiEmbeddingProvider",
        lambda *a, **k: IncompleteProvider(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = _make_repo(
            tmpdir,
            {"module_a.py": "def a(): pass\n", "module_b.py": "def b(): pass\n"},
        )
        session = FakeSession()

        with pytest.raises(ValueError, match="refusing to index partial results"):
            await indexer.index_repository(session)

        assert session.committed is False
        assert len(session.added) == 0


@pytest.mark.asyncio
async def test_index_repository_raises_when_embedding_fails(monkeypatch):
    """Indexing must fail clearly when the embedding provider raises."""

    class FailingProvider:
        dimension = 3072

        async def embed_batch(self, texts, batch_size=50):
            raise RuntimeError("Gemini API failure")

    monkeypatch.setattr(
        "app.services.indexing.indexer.GeminiEmbeddingProvider",
        lambda *a, **k: FailingProvider(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = _make_repo(tmpdir, {"module_a.py": "def a(): pass\n"})
        session = FakeSession()

        with pytest.raises(RuntimeError, match="Gemini API failure"):
            await indexer.index_repository(session)

        assert session.committed is False
        assert len(session.added) == 0


@pytest.mark.asyncio
async def test_index_repository_rejects_none_embedding(monkeypatch):
    """Indexing must reject any chunk whose returned embedding is None."""

    class NoneEmbeddingProvider:
        dimension = 3072

        async def embed_batch(self, texts, batch_size=50):
            return [None] + [[0.01] * 3072 for _ in texts[1:]]

    monkeypatch.setattr(
        "app.services.indexing.indexer.GeminiEmbeddingProvider",
        lambda *a, **k: NoneEmbeddingProvider(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = _make_repo(tmpdir, {"module_a.py": "def a(): pass\n"})
        session = FakeSession()

        with pytest.raises(ValueError, match="Invalid embedding"):
            await indexer.index_repository(session)

        assert session.committed is False
        assert len(session.added) == 0


@pytest.mark.asyncio
async def test_index_repository_success_sets_all_embeddings(monkeypatch):
    """A successful indexing run commits every chunk with a non-None 3072-dim embedding."""
    _good_provider(monkeypatch)

    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = _make_repo(
            tmpdir,
            {"module_a.py": "def a(): pass\n", "module_b.py": "def b(): pass\n"},
        )
        session = FakeSession()

        total, reused = await indexer.index_repository(session)

        assert session.committed is True
        assert reused == 0
        assert len(session.added) == total
        assert all(c.embedding is not None for c in session.added)
        assert all(len(c.embedding) == 3072 for c in session.added)


@pytest.mark.asyncio
async def test_index_repository_reuses_existing_embeddings(monkeypatch):
    """Chunks unchanged since the previous index reuse their stored embedding."""
    _good_provider(monkeypatch)

    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = _make_repo(
            tmpdir,
            {"module_a.py": "def a(): pass\n", "module_b.py": "def b(): pass\n"},
            repo_id=6,
        )
        chunks = indexer.scan_all_chunks()
        first = chunks[0]
        session = FakeSession(
            existing_rows=[
                SimpleNamespace(
                    file_path=first.file_path,
                    content_hash=first.content_hash,
                    embedding=[0.5] * 3072,
                )
            ]
        )

        total, reused = await indexer.index_repository(session)

        assert session.committed is True
        assert reused == 1
        assert total == len(chunks)
        assert len(session.added) == total
        assert all(c.embedding is not None for c in session.added)
        assert all(len(c.embedding) == 3072 for c in session.added)
