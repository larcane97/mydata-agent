from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import create_retriever_tool, Tool

from retriever.RetrieverDispatcher import RetrieverDispatcher
from tool.ToolDispatcher import ToolDispatcher


class MyDataStandardApiSpecPdfTool(ToolDispatcher):
    __path = "data/(221115 수정배포) (2022.10) 금융분야 마이데이터 기술 가이드라인.pdf"

    __name = "mydata_tech_guide_line_retriever_tool"
    __description = ("금융분야 마이데이터 기술 가이드라인 관련 정보를 PDF 문서에서 검색합니다. "
                     "신용정보법에 따른 고객의 개인신용정보 전송요구 및 금융분야 마이데이터서비스 제공과 관련된 세부절차, "
                     "기준 등에 대한 내용을 기술하고 있습니다.")
    __tool: Tool = None

    @classmethod
    def get_tool(cls, /, **kwargs) -> Tool:
        if cls.__tool is not None:
            return cls.__tool
        cls.__tool = cls._get_tool(**kwargs)
        return cls.__tool

    @classmethod
    def _get_tool(cls, **kwargs) -> Tool:
        retriever_dispatcher = kwargs.get("retriever_dispatcher")
        if retriever_dispatcher is None or not isinstance(retriever_dispatcher, RetrieverDispatcher):
            raise AttributeError("retriever_dispatcher가 반드시 필요합니다.")

        pdf_loader = PyPDFLoader(cls.__path)
        retriever = retriever_dispatcher.get_retriever(pdf_loader)

        return create_retriever_tool(
            retriever=retriever,
            name=cls.__name,
            description=cls.__description
        )
