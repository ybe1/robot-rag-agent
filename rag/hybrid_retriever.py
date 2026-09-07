from langchain_core.documents import Document

from rag.bm25_retriever import BM25DocumentRetriever
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf


def document_key(document: Document) -> tuple[str, str]:
    source = str(document.metadata.get("source", "")).replace("\\", "/")
    content = " ".join(document.page_content.split())
    return source, content


class HybridRetriever:
    """使用 RRF 融合向量检索和 BM25 检索结果。"""

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        k: int = 3,
        rrf_constant: int = 60,
        vector_weight: float = 2.0,
        bm25_weight: float = 1.0,
    ):
        if k <= 0:
            raise ValueError("k 必须大于0")
        if rrf_constant < 0:
            raise ValueError("rrf_constant 必须大于等于0")
        if vector_weight <= 0 or bm25_weight <= 0:
            raise ValueError("检索器权重必须大于0")

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.k = k
        self.rrf_constant = rrf_constant
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    @classmethod
    def from_knowledge_base(cls):
        result_k = int(chroma_conf["k"])
        candidate_k = int(chroma_conf["hybrid_candidate_k"])
        vector_store = VectorStoreService("structured")

        return cls(
            vector_retriever=vector_store.get_retriever(k=candidate_k),
            bm25_retriever=BM25DocumentRetriever.from_knowledge_base(
                k=candidate_k
            ),
            k=result_k,
            rrf_constant=int(chroma_conf["rrf_constant"]),
            vector_weight=float(chroma_conf["vector_weight"]),
            bm25_weight=float(chroma_conf["bm25_weight"]),
        )

    def invoke(self, query: str) -> list[Document]:
        ranked_lists = (
            (
                "vector",
                self.vector_weight,
                self.vector_retriever.invoke(query),
            ),
            (
                "bm25",
                self.bm25_weight,
                self.bm25_retriever.invoke(query),
            ),
        )
        fused = {}

        for retriever_name, weight, documents in ranked_lists:
            for rank, document in enumerate(documents, start=1):
                key = document_key(document)
                item = fused.setdefault(
                    key,
                    {
                        "document": document,
                        "score": 0.0,
                        "ranks": {},
                    },
                )
                item["score"] += weight / (self.rrf_constant + rank)
                item["ranks"][retriever_name] = rank

        ranked_items = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:self.k]
        results = []

        for item in ranked_items:
            document = item["document"]
            metadata = dict(document.metadata)
            metadata["retrieval_method"] = "hybrid_rrf"
            metadata["rrf_score"] = item["score"]
            metadata["retrieval_ranks"] = item["ranks"]
            results.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return results
