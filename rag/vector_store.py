import os
import sys

# 将项目根目录加入 sys.path，便于在 VSCode 中直接运行脚本时导入 utils / model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embed_model
from rag.document_splitter import StructuredDocumentSplitter
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self, profile_name: str | None = None):
        self.profile_name = profile_name or chroma_conf.get(
            'active_profile',
            'baseline'
        )
        profiles = chroma_conf.get('profiles', {})

        if self.profile_name not in profiles:
            available_profiles = ', '.join(profiles) or '无'
            raise ValueError(
                f'未知向量库配置：{self.profile_name}；'
                f'可用配置：{available_profiles}'
            )

        self.profile = profiles[self.profile_name]

        self.vector_store=Chroma(
            collection_name=self.profile['collection_name'],
            embedding_function=embed_model,
            persist_directory=self.profile['persist_directory']
        )

        splitter_name = self.profile['splitter']

        if splitter_name == 'recursive':
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chroma_conf['chunk_size'],
                chunk_overlap=chroma_conf['chunk_overlap'],
                separators=chroma_conf['separators'],
                length_function=len,
            )
        elif splitter_name == 'structured':
            self.splitter = StructuredDocumentSplitter(
                max_chunk_size=chroma_conf['chunk_size'],
                chunk_overlap=chroma_conf['chunk_overlap'],
            )
        else:
            raise ValueError(
                f'配置 {self.profile_name} 使用了未知切分器：'
                f'{splitter_name}'
            )

        logger.info(
            f'[向量库]当前配置：{self.profile_name}，'
            f'切分器：{splitter_name}，'
            f'存储目录：{self.profile["persist_directory"]}'
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={'k':chroma_conf['k']})

    def load_documents(self):
        def check_md5_hex(md5_for_check:str):
            md5_store_path = get_abs_path(self.profile['md5_hex_store'])

            if not os.path.exists(md5_store_path):
                open(md5_store_path,'w',encoding='utf-8').close()
                return False
        
            with open(md5_store_path,'r',encoding='utf-8') as f:
                for line in f.readlines():
                    line=line.strip()
                    if line==md5_for_check:
                        return True
        
                return False
        
        def save_md5_hex(md5_for_check:str):
            md5_store_path = get_abs_path(self.profile['md5_hex_store'])

            with open(md5_store_path,'a',encoding='utf-8') as f:
                f.write(md5_for_check+'\n')
        
        def get_file_documents(read_path:str):
            if read_path.endswith('pdf'):
                return pdf_loader(read_path)
        
            if read_path.endswith('txt'):
                return txt_loader(read_path)
        
            return []

        allowed_files_path:list[str]=listdir_with_allowed_type(
            get_abs_path(chroma_conf['data_path']),
            tuple(chroma_conf['allow_knowledge_file_type'])
        )

        for path in allowed_files_path:
            md5_hex=get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f'[加载知识库]{path}内容已经存在知识库内，跳过')
                continue

            try:
                documents:list[Document]=get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_documents:list[Document]=self.splitter.split_documents(documents)

                if not split_documents:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_documents)

                save_md5_hex(md5_hex)

                logger.info(
                    f'[加载知识库]{path}内容加载成功，'
                    f'共生成{len(split_documents)}个结构化分片'
                )

            except Exception as e:
                logger.error(f'[加载知识库]{path}加载失败：{str(e)}',exc_info=True)
                continue


if __name__ == '__main__':
    vs=VectorStoreService()
    vs.load_documents()

    retriever=vs.get_retriever()
    
    res=retriever.invoke('迷路')
    for r in res:
        print(r.page_content)
        print('-'*30)



    
