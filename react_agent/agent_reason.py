from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv

load_dotenv()
llm  = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@tool
def get_system_time(format: str="%Y-%m-%d %H:%M:%S"):
    """ Returns the current system time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic" )



tools = [search_tool , get_system_time]

agent_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools , llm=llm , prompt=agent_prompt)