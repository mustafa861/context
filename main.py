import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper
from agents.run import RunConfig
from agents.tool import function_tool

load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@dataclass
class UserInfo:
    name: str
    age: int
    rollno: int
    location: str = "Pakistan"

@function_tool
async def fetch_userinfo_from_database(wrapper: RunContextWrapper[UserInfo]) -> str:
    return f"""User {wrapper.context.name} is {wrapper.context.age} years old, 
and the roll no of user is {wrapper.context.rollno}, 
and his current location is {wrapper.context.location}."""

async def main():
    user_info = UserInfo(name="Muhammad Mustafa", age=12, rollno=17)

    agent = Agent[UserInfo](
        name="Assistant",
        tools=[fetch_userinfo_from_database],
        
    )

    result = await Runner.run(
        starting_agent=agent,
        input="Use the fetch_userinfo_from_database tool to get all information of user.",
        run_config=run_config,
        context=user_info
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
