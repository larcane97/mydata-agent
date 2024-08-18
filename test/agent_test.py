from unittest.mock import Mock

from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain_core.tools import Tool

from chat_model.DolphinLlama3_8bChatModel import DolphinLlama3_8bChatModel
from chat_model.XionicKoLlama3_70bChatModel import XionicKoLlama3_70bChatModel
from config.Config import Config

"""
parameter 수가 적은 chatModel의 경우 tools을 제대로 사용하지 못 하는 경우가 존재하기 때문에 새로운 chatModel을 사용하기 전에 반드시 테스트해봐야 한다.
"""


def test__XionicKoLlama3_70bChatModel__agent():
    """
    XinoicKoLlam3_70B chatModel Agent가 정상적으로 tool을 호출하는지 여부를 판단하기 위한 테스트
    """
    chat_model = XionicKoLlama3_70bChatModel.get_chat_model()

    mocking_func = Mock(return_value="마이데이터는 개발자 임문경이 정의하였습니다.")

    tool = Tool(
        name="mydata_tech_guide_line_retriever_tool",
        description=("금융분야 마이데이터 기술 가이드라인 관련 정보를 PDF 문서에서 검색합니다. "
                     "신용정보법에 따른 고객의 개인신용정보 전송요구 및 금융분야 마이데이터서비스 제공과 관련된 세부절차, "
                     "기준 등에 대한 내용을 기술하고 있습니다."),
        func=mocking_func
    )

    agent = create_json_chat_agent(
        llm=chat_model,
        tools=[tool],
        prompt=Config.prompt()
    )

    executor = AgentExecutor(
        agent=agent,
        tools=[tool],
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    response = executor.invoke({"input": "마이데이터를 정의한 사람은 누구입니까?"})

    assert mocking_func.called
    assert response is not None
    assert "임문경" in response["output"]


def test__DolphinLlam3_8BChatModel__agent():
    """
    DolphinLlam3_8BChatModel chatModel Agent가 정상적으로 tool을 호출하는지 여부를 판단하기 위한 테스트
    """
    chat_model = DolphinLlama3_8bChatModel.get_chat_model()

    mocking_func = Mock(return_value="마이데이터는 개발자 임문경이 정의하였습니다.")

    tool = Tool(
        name="mydata_tech_guide_line_retriever_tool",
        description=("금융분야 마이데이터 기술 가이드라인 관련 정보를 PDF 문서에서 검색합니다. "
                     "신용정보법에 따른 고객의 개인신용정보 전송요구 및 금융분야 마이데이터서비스 제공과 관련된 세부절차, "
                     "기준 등에 대한 내용을 기술하고 있습니다."),
        func=mocking_func
    )

    agent = create_json_chat_agent(
        llm=chat_model,
        tools=[tool],
        prompt=Config.prompt()
    )

    executor = AgentExecutor(
        agent=agent,
        tools=[tool],
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    response = executor.invoke({"input": "마이데이터를 정의한 사람은 누구입니까?"})

    print("response : ", response)

    assert mocking_func.called
    assert response is not None
    assert "임문경" in response["output"]
