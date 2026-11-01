"""Build and search a local vector index of travel-company and public policy PDFs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("policy_service.index")


@dataclass(frozen=True)
class Passage:
    document_id: str
    document_title: str
    issuer: str
    category: str
    family: str
    source_url: str
    published_date: str | None
    effective_date: str | None
    validity_note: str
    retrieved_at: str
    page_number: int
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "issuer": self.issuer,
            "category": self.category,
            "family": self.family,
            "source_url": self.source_url,
            "published_date": self.published_date,
            "effective_date": self.effective_date,
            "validity_note": self.validity_note,
            "retrieved_at": self.retrieved_at,
            "page_numbers": [self.page_number],
            "excerpt": self.excerpt,
        }


def _clean_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[\t\r\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _split_text(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    """Split extracted page text into bounded, overlapping chunks."""
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_characters:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_characters)
        if end < len(text):
            boundary = max(
                text.rfind(". ", start + max_characters // 2, end),
                text.rfind("; ", start + max_characters // 2, end),
                text.rfind(" ", start + max_characters // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_characters)
    return chunks


class PolicyIndex:
    def __init__(
        self,
        *,
        model: SentenceTransformer,
        vector_index: Any,
        passages: list[Passage],
        document_count: int,
        warnings: list[str],
        minimum_similarity: float,
        result_limit: int,
        model_name: str,
    ) -> None:
        self.model = model
        self.vector_index = vector_index
        self.passages = passages
        self.document_count = document_count
        self.warnings = warnings
        self.minimum_similarity = minimum_similarity
        self.result_limit = result_limit
        self.model_name = model_name

    @classmethod
    def load(cls, service_dir: Path, config: dict[str, Any]) -> "PolicyIndex":
        manifest_path = service_dir / config.get("manifestFile", "sources.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents = manifest.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError("The policy source manifest contains no documents.")

        passages: list[Passage] = []
        warnings: list[str] = []
        readable_documents = 0
        for document in documents:
            if not isinstance(document, dict):
                warnings.append("Skipped a malformed source-manifest entry.")
                continue
            filename = document.get("filename")
            pdf_path = service_dir / "documents" / str(filename or "")
            if not filename or not pdf_path.is_file():
                warnings.append(f"Missing PDF: {filename or 'unnamed source'}")
                continue
            try:
                reader = PdfReader(str(pdf_path), strict=False)
                before_count = len(passages)
                for page_index, page in enumerate(reader.pages, start=1):
                    page_text = _clean_text(page.extract_text() or "")
                    for excerpt in _split_text(
                        page_text,
                        max_characters=max(300, int(config.get("chunkCharacters", 1200))),
                        overlap_characters=max(0, int(config.get("chunkOverlapCharacters", 180))),
                    ):
                        passages.append(Passage(
                            document_id=str(document.get("id", pdf_path.stem)),
                            document_title=str(document.get("title", pdf_path.stem)),
                            issuer=str(document.get("issuer", "Unknown issuer")),
                            category=str(document.get("category", "policy")),
                            family=str(document.get("family", "Policy source")),
                            source_url=str(document.get("source_url", "")),
                            published_date=document.get("published_date"),
                            effective_date=document.get("effective_date"),
                            validity_note=str(document.get("validity_note", "")),
                            retrieved_at=str(manifest.get("retrieved_at", "")),
                            page_number=page_index,
                            excerpt=excerpt,
                        ))
                if len(passages) > before_count:
                    readable_documents += 1
                else:
                    warnings.append(f"No extractable text: {filename}")
            except Exception as exc:
                warnings.append(f"Could not read {filename}: {type(exc).__name__}")
                LOGGER.warning("Could not extract policy PDF %s (%s)", filename, type(exc).__name__)

        if not passages:
            raise ValueError("No policy PDF text could be extracted from the configured corpus.")

        model_name = str(config.get("modelName", "sentence-transformers/all-MiniLM-L6-v2"))
        model = SentenceTransformer(model_name, device=str(config.get("device", "cpu")))
        # Encode bounded groups so startup memory use stays predictable for the
        # bundled corpus, even on machines without an accelerator.
        batch_size = max(1, min(32, int(config.get("embeddingBatchSize", 16))))
        vector_batches = []
        excerpts = [passage.excerpt for passage in passages]
        for start in range(0, len(excerpts), batch_size):
            vector_batches.append(model.encode_document(
                excerpts[start:start + batch_size],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            ))
        vectors = np.ascontiguousarray(np.concatenate(vector_batches, axis=0), dtype=np.float32)
        # Import FAISS only after PyTorch has completed model inference. On
        # macOS, importing FAISS first can initialize a conflicting native
        # OpenMP runtime and crash SentenceTransformer during tokenization.
        import faiss

        vector_index = faiss.IndexFlatIP(vectors.shape[1])
        vector_index.add(vectors)

        return cls(
            model=model,
            vector_index=vector_index,
            passages=passages,
            document_count=readable_documents,
            warnings=warnings,
            minimum_similarity=float(config.get("minimumSimilarity", 0.22)),
            result_limit=max(1, min(2, int(config.get("resultLimit", 2)))),
            model_name=model_name,
        )

    def search(self, question: str) -> list[dict[str, Any]]:
        query_vector = self.model.encode_query(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        candidate_count = min(len(self.passages), max(self.result_limit * 5, self.result_limit))
        scores, indices = self.vector_index.search(query_vector, candidate_count)

        matches: list[dict[str, Any]] = []
        seen_pages: set[tuple[str, int]] = set()
        for score, passage_index in zip(scores[0], indices[0]):
            if passage_index < 0 or float(score) < self.minimum_similarity:
                continue
            passage = self.passages[int(passage_index)]
            page_key = (passage.document_id, passage.page_number)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            result = passage.as_dict()
            result["similarity_score"] = float(score)
            matches.append(result)
            LOGGER.info(json.dumps({
                "event": "policy_retrieval_match",
                "document_id": passage.document_id,
                "page_number": passage.page_number,
                "similarity_score": round(float(score), 4),
            }, ensure_ascii=False))
            if len(matches) >= self.result_limit:
                break
        if not matches:
            best_score = float(scores[0][0]) if candidate_count else None
            LOGGER.info(json.dumps({
                "event": "policy_retrieval_no_match",
                "best_similarity_score": round(best_score, 4) if best_score is not None else None,
                "minimum_similarity": self.minimum_similarity,
            }, ensure_ascii=False))
        return matches

