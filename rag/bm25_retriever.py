import re

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.document_splitter import StructuredDocumentSplitter
from utils.config_handler import chroma_conf
from utils.file_handler import (
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.path_tool import get_abs_path


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """将中英文文本转换为适合 BM25 的小写词元。"""
    normalized_text = " ".join(TOKEN_PATTERN.findall(text.lower()))
    return [
        token.strip()
        for token in jieba.lcut(normalized_text)
        if token.strip()
    ]


class BM25DocumentRetriever:
    """基于本地知识文件和结构化分片的 BM25 检索器。"""

    def __init__(self, documents: list[Document], k: int = 3):
        if not documents:
            raise ValueError("BM25 检索器需要至少一个文档分片")
        if k <= 0:
            raise ValueError("k 必须大于0")

        self.documents = documents
        self.k = k
        self.tokenized_corpus = [
            tokenize(document.page_content)
            for document in documents
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_knowledge_base(cls, k: int | None = None):
        splitter = StructuredDocumentSplitter(
            max_chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
        )
        paths = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        documents = []

        for path in paths:
            if path.lower().endswith(".pdf"):
                loaded_documents = pdf_loader(path)
            elif path.lower().endswith(".txt"):
                loaded_documents = txt_loader(path)
            else:
                continue

            documents.extend(splitter.split_documents(loaded_documents))

        return cls(documents, k=k or int(chroma_conf["k"]))

    def invoke(self, query: str) -> list[Document]:
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: scores[index],
            reverse=True,
        )[:self.k]

        return [
            self.documents[index]
            for index in ranked_indices
            if scores[index] > 0
        ]
