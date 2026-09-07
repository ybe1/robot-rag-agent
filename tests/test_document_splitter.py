import pytest
from langchain_core.documents import Document

from rag.document_splitter import StructuredDocumentSplitter


def test_numbered_entries_are_split_without_mixing_answers():
    document = Document(
        page_content=(
            "使用说明\n"
            "1. 无法开机怎么办？检查电池和电源按键。\n"
            "2. 找不到充电座怎么办？清理障碍物和回充传感器。"
        ),
        metadata={"source": "故障排除.txt"},
    )

    chunks = StructuredDocumentSplitter().split_documents([document])

    assert len(chunks) == 3
    assert "检查电池" in chunks[1].page_content
    assert "找不到充电座" not in chunks[1].page_content
    assert "清理障碍物" in chunks[2].page_content
    assert chunks[2].metadata["source"] == "故障排除.txt"
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]


def test_empty_documents_are_ignored():
    chunks = StructuredDocumentSplitter().split_documents(
        [Document(page_content="  ")]
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_invalid_splitter_parameters_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        StructuredDocumentSplitter(chunk_size, overlap)
