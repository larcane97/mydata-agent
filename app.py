import asyncio
import logging
import uuid
from asyncio import Future
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_json_chat_agent, AgentExecutor
from starlette.middleware.cors import CORSMiddleware

from config.Config import Config

logger = logging.getLogger("mydata-agent")

# set env
load_dotenv()

# set agent
llama3_agent = create_json_chat_agent(
    llm=Config.chat_model(),
    tools=Config.tools(),
    prompt=Config.prompt())

executor = AgentExecutor(
    agent=llama3_agent,
    tools=Config.tools(),
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
MAX_DELAY_SECONDS = 1  # 100ms
MAX_PENDING_REQUESTS = 10

request_queue = []
pending_responses: Dict[str, Future] = {}


async def process_batch():
    while True:
        await asyncio.sleep(MAX_DELAY_SECONDS)
        if request_queue:
            batch_data = request_queue[:]
            request_queue.clear()
            logger.info(f"process batch.. current queue size : {len(batch_data)}")

            responses = executor.batch([{"input": request["user_input"]} for request in batch_data])
            for request, response in zip(batch_data, responses):
                request_id = request["id"]
                if request_id in pending_responses:
                    pending_responses[request_id].set_result(response["output"])
                    del pending_responses[request_id]


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batch())


@app.post("/chat")
async def predict(user_input: str):
    if len(request_queue) > MAX_PENDING_REQUESTS:
        return "현재 요청이 너무 많습니다. 잠시 후 다시 시도해주세요."

    request_id = str(uuid.uuid4())
    future = asyncio.Future()

    pending_responses[request_id] = future
    request_queue.append({"id": request_id, "user_input": user_input})

    response = await future
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
