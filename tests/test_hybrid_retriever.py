from langchain_core.documents import Document

from rag.hybrid_retriever import HybridRetriever, document_key


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


def make_document(content, source="知识库.txt"):
    return Document(page_content=content, metadata={"source": source})


def test_rrf_merges_duplicates_and_preserves_rank_metadata():
    vector_first = make_document("向量结果")
    shared = make_document("共同结果")
    bm25_only = make_document("关键词结果")
    retriever = HybridRetriever(
        vector_retriever=FakeRetriever([vector_first, shared]),
        bm25_retriever=FakeRetriever([shared, bm25_only]),
        k=3,
        vector_weight=2.0,
        bm25_weight=1.0,
    )

    results = retriever.invoke("测试问题")

    assert [document.page_content for document in results] == [
        "共同结果",
        "向量结果",
        "关键词结果",
    ]
    assert len(results) == 3
    assert results[0].metadata["retrieval_method"] == "hybrid_rrf"
    assert results[0].metadata["retrieval_ranks"] == {
        "vector": 2,
        "bm25": 1,
    }


def test_document_key_normalizes_path_and_whitespace():
    first = make_document("同一  段\n内容", "data\\知识库.txt")
    second = make_document("同一 段 内容", "data/知识库.txt")

    assert document_key(first) == document_key(second)
