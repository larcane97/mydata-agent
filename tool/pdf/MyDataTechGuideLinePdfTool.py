from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import create_retriever_tool, Tool

from retriever.RetrieverDispatcher import RetrieverDispatcher
from tool.ToolDispatcher import ToolDispatcher


class MyDataTechGuideLinePdfTool(ToolDispatcher):
    __path = "data/(수정게시) 금융분야 마이데이터 표준 API 규격 v1.pdf"

    __name = "mydata_standard_api_spec_retriever_tool"
    __description = ("금융분야 마이데이터 표준 API 규격 관련 정보를 PDF 문서에서 검색합니다. "
                     "마이데이터사업자가 신용정보제공 이용자등으로부터 개인 신용정보를 안전하고 신뢰할 수 있는 방식으로 제공받기 위한 API 규격을 기술하고 있습니다.")
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
