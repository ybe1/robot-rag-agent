import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts

class RagSummarizeService(object):
    def __init__(self, profile_name: str = "structured"):
        self.vector_store=VectorStoreService(profile_name)
        self.retriever=self.vector_store.get_retriever()
        self.prompt_text=load_rag_prompts()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model
        self.chain=self._init_chain()

    def _init_chain(self):
        chain=self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self,query:str)->list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize_with_sources(self, query: str) -> dict:
        context_docs=self.retriever_docs(query)

        sources = []
        context_parts = []

        for index, doc in enumerate(context_docs, start=1):
            source_path = str(doc.metadata.get("source", "未知来源"))
            source = Path(source_path).name
            page = doc.metadata.get("page")
            citation = {
                "id": index,
                "source": source,
                "page": page + 1 if isinstance(page, int) else None,
                "content": doc.page_content.strip(),
            }
            sources.append(citation)
            page_text = f"，第{citation['page']}页" if citation["page"] else ""
            context_parts.append(
                f"[资料{index}] 来源：{source}{page_text}\n"
                f"{citation['content']}"
            )

        context = "\n\n".join(context_parts) or "没有检索到参考资料"

        answer = self.chain.invoke(
            {
                'input':query,
                'context':context,
            }
        )

        answer = answer.strip()
        cited_ids = {
            int(match)
            for match in re.findall(r"\[资料(\d+)\]", answer)
        }
        cited_sources = [
            source
            for source in sources
            if source["id"] in cited_ids
        ]

        return {
            "answer": answer,
            "sources": cited_sources,
        }

    def rag_summarize(self,query:str)->str:
        return self.rag_summarize_with_sources(query)["answer"]

if __name__ == "__main__":
    rag=RagSummarizeService()

    print(rag.rag_summarize('小户型适合哪些扫地机器人？'))
