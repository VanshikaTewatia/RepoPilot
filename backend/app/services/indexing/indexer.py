"""Repository indexing pipeline orchestrator."""

from pathlib import Path
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.logging import logger
from app.db.models.code_chunk import CodeChunk
from app.db.models.repository import Repository
from app.services.indexing.file_discovery import detect_language, discover_source_files
from app.services.indexing.parsers import get_parser_for_language
from app.services.embeddings.gemini import GeminiEmbeddingProvider


class RepositoryIndexer:
    """Orchestrates file discovery, syntax-aware AST parsing, and chunk generation."""

    def __init__(self, repo_path: Path | str, repo_id: int):
        self.repo_root = Path(repo_path).resolve()
        self.repo_id = repo_id

    def extract_chunks_from_file(self, file_path: Path) -> List[CodeChunk]:
        """Extract syntax-aware CodeChunk models from a single source file."""
        rel_path = str(file_path.relative_to(self.repo_root)).replace("\\", "/")
        language = detect_language(file_path) or "python"
        parser = get_parser_for_language(language)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        symbols = parser.parse(source_code, file_path=rel_path)
        chunks: List[CodeChunk] = []

        for sym in symbols:
            chunk = CodeChunk(
                repository_id=self.repo_id,
                file_path=rel_path,
                language=language,
                symbol_name=sym.name,
                symbol_type=sym.symbol_type,
                start_line=sym.start_line,
                end_line=sym.end_line,
                source_code=sym.source_code,
                content_hash=sym.content_hash,
                chunk_metadata={
                    "docstring": sym.docstring,
                    "parent_symbol": sym.parent_symbol,
                    **sym.metadata,
                },
            )
            chunks.append(chunk)

        return chunks

    def scan_all_chunks(self) -> List[CodeChunk]:
        """Discover and parse all valid source files in the repository."""
        all_chunks: List[CodeChunk] = []
        for file_path in discover_source_files(self.repo_root):
            chunks = self.extract_chunks_from_file(file_path)
            all_chunks.extend(chunks)
        return all_chunks

    async def index_repository(self, db: AsyncSession) -> Tuple[int, int]:
        """Index or update repository code chunks in the database.

        Returns: (total_chunks_created, total_chunks_skipped)
        """
        all_new_chunks = self.scan_all_chunks()

        # Fetch existing chunk hashes for this repository to avoid redundant re-embedding
        existing_result = await db.execute(
            select(CodeChunk.file_path, CodeChunk.content_hash, CodeChunk.embedding).where(
                CodeChunk.repository_id == self.repo_id
            )
        )
        existing_map = {
            (row.file_path, row.content_hash): row.embedding
            for row in existing_result.all()
        }

        # Clear existing chunks for clean update
        await db.execute(
            delete(CodeChunk).where(CodeChunk.repository_id == self.repo_id)
        )

        # Separate chunks into those that can reuse embeddings vs those needing new embeddings
        reuse_chunks: List[CodeChunk] = []
        embed_chunks: List[CodeChunk] = []

        for chunk in all_new_chunks:
            existing_emb = existing_map.get((chunk.file_path, chunk.content_hash))
            if existing_emb is not None:
                chunk.embedding = existing_emb
                reuse_chunks.append(chunk)
            else:
                embed_chunks.append(chunk)

        # Generate embeddings for new/modified chunks in batch
        if embed_chunks:
            try:
                provider = GeminiEmbeddingProvider()
                texts = [c.source_code for c in embed_chunks]
                vectors = await provider.embed_batch(texts)
            except Exception as e:
                logger.error(f"Embedding generation failed for repository {self.repo_id}: {e}")
                raise

            if len(vectors) != len(embed_chunks):
                raise ValueError(
                    f"Embedding provider returned {len(vectors)} vectors for "
                    f"{len(embed_chunks)} input chunks; refusing to index partial results."
                )
            for chunk, vector in zip(embed_chunks, vectors):
                if vector is None or len(vector) != provider.dimension:
                    raise ValueError(
                        f"Invalid embedding for chunk {chunk.symbol_name!r} in "
                        f"{chunk.file_path}: expected a {provider.dimension}-dimensional vector."
                    )
                chunk.embedding = vector
            logger.info(
                f"Generated {len(embed_chunks)} embeddings for repository {self.repo_id}."
            )

        # Never persist chunks without embeddings
        missing = [c for c in all_new_chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"Refusing to index {len(missing)} chunks with missing embeddings "
                f"(e.g. {missing[0].file_path}:{missing[0].symbol_name})."
            )

        for chunk in all_new_chunks:
            db.add(chunk)

        await db.commit()
        created_count = len(embed_chunks)
        reused_count = len(reuse_chunks)
        logger.info(
            f"Indexed repository {self.repo_id}: {len(all_new_chunks)} total chunks "
            f"({created_count} new/modified, {reused_count} reused)."
        )
        return len(all_new_chunks), reused_count

