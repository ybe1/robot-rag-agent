from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


class FakeChain:
    def __init__(self, answer):
        self.answer = answer
        self.last_input = None

    def invoke(self, chain_input):
        self.last_input = chain_input
        return self.answer


def make_service(documents, answer):
    service = RagSummarizeService.__new__(RagSummarizeService)
    service.retriever = FakeRetriever(documents)
    service.chain = FakeChain(answer)
    return service


def test_only_sources_cited_in_answer_are_returned():
    service = make_service(
        [
            Document(page_content="无关资料", metadata={"source": "a.txt"}),
            Document(
                page_content="检查充电触点。",
                metadata={"source": "data/故障排除.pdf", "page": 1},
            ),
        ],
        "请检查充电触点。[资料2]",
    )

    result = service.rag_summarize_with_sources("无法充电")

    assert result["answer"] == "请检查充电触点。[资料2]"
    assert result["sources"] == [
        {
            "id": 2,
            "source": "故障排除.pdf",
            "page": 2,
            "content": "检查充电触点。",
        }
    ]


def test_refusal_does_not_expose_irrelevant_sources():
    service = make_service(
        [Document(page_content="机器人资料", metadata={"source": "知识库.txt"})],
        "现有资料不足，无法可靠回答这个问题。",
    )

    result = service.rag_summarize_with_sources("如何制作红烧肉？")

    assert result["sources"] == []


def test_empty_retrieval_is_described_to_the_model():
    service = make_service([], "现有资料不足，无法可靠回答这个问题。")

    service.rag_summarize_with_sources("未知问题")

    assert service.chain.last_input["context"] == "没有检索到参考资料"
