import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 匹配行首的编号条目，例如：
# 1. 故障现象……
# 15. **电动拖布不旋转怎么办？**
ENTRY_PATTERN = re.compile(
    r"(?=^\s*\d+\.\s)",
    flags=re.MULTILINE,
)


class StructuredDocumentSplitter:
    """优先按编号条目切分，超长条目再按字符切分。"""

    def __init__(
        self,
        max_chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size 必须大于0")

        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            raise ValueError("chunk_overlap 必须大于等于0且小于max_chunk_size")

        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                ".",
                " ",
                "",
            ],
            length_function=len,
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        chunks = []

        for document in documents:
            chunks.extend(self._split_document(document))

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["split_method"] = "structured"

        return chunks

    def _split_document(
        self,
        document: Document,
    ) -> list[Document]:
        text = document.page_content.strip()

        if not text:
            return []

        entries = [
            entry.strip()
            for entry in ENTRY_PATTERN.split(text)
            if entry.strip()
        ]

        chunks = []

        for entry in entries:
            entry_document = Document(
                page_content=entry,
                metadata=dict(document.metadata),
            )

            if len(entry) <= self.max_chunk_size:
                chunks.append(entry_document)
            else:
                chunks.extend(
                    self.fallback_splitter.split_documents(
                        [entry_document]
                    )
                )

        return chunks
