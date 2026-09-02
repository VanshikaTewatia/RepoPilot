"""Code-aware semantic retrieval and RAG question answering."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from google import genai
from app.core.config import settings
from app.core.logging import logger
from app.db.models.code_chunk import CodeChunk
from app.services.embeddings.gemini import GeminiEmbeddingProvider


class RetrievalError(RuntimeError):
    """Raised when semantic retrieval cannot be performed (e.g. query embedding fails)."""


@dataclass
class RetrievedChunk:
    """Represents a retrieved code chunk with similarity score and citation."""

    file_path: str
    symbol_name: Optional[str]
    symbol_type: str
    start_line: int
    end_line: int
    source_code: str
    similarity_score: float

    @property
    def citation(self) -> str:
        """Formatted source citation e.g. 'src/calculator.py:12-18'."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class CodeRetriever:
    """Performs semantic code retrieval and contextual question answering."""

    def __init__(
        self,
        embedding_provider: Optional[GeminiEmbeddingProvider] = None,
    ):
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider()

    async def retrieve_chunks(
        self,
        query: str,
        repository_id: int,
        top_k: Optional[int] = None,
        file_prefix: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> List[RetrievedChunk]:
        """Perform pgvector cosine similarity search to find relevant code chunks.

        ``top_k`` defaults to ``settings.vector_top_k`` when omitted (rather
        than a hardcoded literal), so the retrieval limit is configurable in
        one place. ``file_prefix``, when given, scopes results to chunks
        whose ``file_path`` falls under that project root (e.g. "frontend"
        in a monorepo) -- omit it (the default) to search the whole
        repository exactly as before; existing callers are unaffected.
        """
        if not query.strip() or not db:
            return []

        effective_top_k = top_k if top_k is not None else settings.vector_top_k

        # Generate dense query embedding
        try:
            query_vector = await self.embedding_provider.embed_text(query)
        except Exception as e:
            logger.error(
                f"Query embedding generation failed for repository {repository_id}: {e}"
            )
            raise RetrievalError(f"Semantic retrieval unavailable: {e}") from e

        # Query pgvector for closest chunks by cosine distance
        distance_expr = CodeChunk.embedding.cosine_distance(query_vector)
        conditions = [
            CodeChunk.repository_id == repository_id,
            CodeChunk.embedding.isnot(None),
        ]
        if file_prefix and file_prefix != ".":
            conditions.append(CodeChunk.file_path.startswith(file_prefix.rstrip("/") + "/"))

        stmt = (
            select(CodeChunk, distance_expr.label("distance"))
            .where(*conditions)
            .order_by("distance")
            .limit(effective_top_k)
        )

        result = await db.execute(stmt)
        rows = result.all()

        retrieved: List[RetrievedChunk] = []
        for chunk, distance in rows:
            # Cosine distance to similarity: 1 - distance
            similarity = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
            retrieved.append(
                RetrievedChunk(
                    file_path=chunk.file_path,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    source_code=chunk.source_code,
                    similarity_score=round(similarity, 4),
                )
            )

        return retrieved

    async def answer_question(
        self,
        query: str,
        repository_id: int,
        top_k: Optional[int] = None,
        file_prefix: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant code context and generate a grounded answer.

        See ``retrieve_chunks`` for ``top_k``/``file_prefix`` semantics.
        """
        try:
            chunks = await self.retrieve_chunks(
                query=query,
                repository_id=repository_id,
                top_k=top_k,
                file_prefix=file_prefix,
                db=db,
            )
        except RetrievalError as e:
            logger.error(f"Semantic retrieval failed for repository {repository_id}: {e}")
            return {
                "answer": f"Could not perform semantic retrieval: {e}",
                "retrieved_chunks": [],
                "citations": [],
            }

        if not chunks:
            return {
                "answer": "No relevant code chunks found for this query in the repository.",
                "retrieved_chunks": [],
                "citations": [],
            }

        # Build context prompt
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"### File: {c.file_path} (Lines {c.start_line}-{c.end_line})\n"
                f"Symbol: {c.symbol_name or 'block'} ({c.symbol_type})\n"
                f"```\n{c.source_code}\n```"
            )
        context_str = "\n\n".join(context_parts)

        system_instruction = (
            "You are RepoPilot, an expert AI software engineer. Answer the user's question "
            "using ONLY the provided code snippets. Always cite specific files and exact line "
            "ranges in format `filename.py:start-end`. Do not expose internal chain-of-thought; "
            "keep your response concise, clear, and direct."
        )

        prompt = (
            f"Context Code Chunks:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a clear answer referencing the specific files and line numbers."
        )

        answer_text = ""
        try:
            if settings.gemini_api_key:
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model=settings.gemini_model_name,
                    contents=prompt,
                    config={"system_instruction": system_instruction},
                )
                answer_text = response.text or ""
            else:
                # Simulated answer in test environment
                citations_preview = ", ".join(c.citation for c in chunks)
                answer_text = (
                    f"Based on the codebase analysis ({citations_preview}), "
                    f"here is the explanation for: '{query}'."
                )
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            answer_text = f"Retrieved relevant code chunks but encountered LLM generation error: {e}"

        return {
            "answer": answer_text,
            "retrieved_chunks": [
                {
                    "file_path": c.file_path,
                    "symbol_name": c.symbol_name,
                    "symbol_type": c.symbol_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "source_code": c.source_code,
                    "similarity_score": c.similarity_score,
                    "citation": c.citation,
                }
                for c in chunks
            ],
            "citations": [c.citation for c in chunks],
        }
