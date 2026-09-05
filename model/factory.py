import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

from utils.config_handler import rag_conf


# factory.py 位于 model/ 下，因此向上两级得到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 本地开发时读取 .env；已存在的环境变量优先。
load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "未配置 DASHSCOPE_API_KEY。"
            "请在项目根目录创建 .env 并填写密钥，"
            "或在运行环境中设置该变量。"
        )

    return api_key


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key=get_api_key(),
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=get_api_key(),
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()