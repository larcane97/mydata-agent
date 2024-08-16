from typing import List

from langchain import hub
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from chat_model.DolphinLlama3_8bChatModel import DolphinLlama3_8bChatModel
from chat_model.XionicKoLlama3_70bChatModel import XionicKoLlama3_70bChatModel
from embedding.MultiLingualE5LargeEmbedding import MultiLingualE5LargeEmbedding
from retriever.EnsembleFaissBM25RetrieverDispatcher import EnsembleFaissBM25RetrieverDispatcher
from retriever.RaptorFaissRetrieverDispatcher import RaptorFaissRetrieverDispatcher
from retriever.RetrieverDispatcher import RetrieverDispatcher
from tool.pdf.MyDataStandardApiSpecPdfTool import MyDataStandardApiSpecPdfTool
from tool.pdf.MyDataTechGuideLinePdfTool import MyDataTechGuideLinePdfTool


class Config:

    @classmethod
    def embedding_model(cls) -> Embeddings:
        return MultiLingualE5LargeEmbedding.get_embedding()

    @classmethod
    def tools(cls) -> List[Tool]:
        tools = [
            MyDataStandardApiSpecPdfTool.get_tool(
                retriever_dispatcher=cls.get_retriever_dispatcher()
            ),
            MyDataTechGuideLinePdfTool.get_tool(
                retriever_dispatcher=cls.get_retriever_dispatcher()
            )
        ]

        return tools

    @classmethod
    def get_retriever_dispatcher(cls) -> RetrieverDispatcher:
        return EnsembleFaissBM25RetrieverDispatcher(
            text_splitter=cls.text_splitters(),
            embedding_model=cls.embedding_model(),
        )

    @classmethod
    def chat_model(cls) -> BaseChatModel:
        return XionicKoLlama3_70bChatModel.get_chat_model()

    @classmethod
    def text_splitters(cls) -> TextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100)

    @classmethod
    def prompt(cls) -> ChatPromptTemplate:
        return hub.pull("teddynote/react-chat-json-korean")
