from typing import Union, List

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import TextSplitter

from retriever.RetrieverDispatcher import RetrieverDispatcher


class FaissRetrieverDispatcher(RetrieverDispatcher):
    text_splitter: TextSplitter
    embedding_model: Embeddings

    def __init__(self, embedding_model: Embeddings, text_splitter: TextSplitter = None):
        if embedding_model is None:
            raise AttributeError("embedding_model이 반드시 필요합니다.")

        self.text_splitter = text_splitter
        self.embedding_model = embedding_model

    def get_retriever(self, docs: Union[BaseLoader, List[Document]], **kwargs) -> BaseRetriever:
        if isinstance(docs, BaseLoader):
            if self.embedding_model is None or not isinstance(self.text_splitter, TextSplitter):
                docs = docs.load()
            else:
                docs = docs.load_and_split(self.text_splitter)

        faiss_vector_store = FAISS.from_documents(docs, embedding=self.embedding_model)

        return faiss_vector_store.as_retriever(**kwargs)
