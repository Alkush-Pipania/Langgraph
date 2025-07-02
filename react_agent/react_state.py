import operator
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv
from langchain_core.agents import AgentAction , AgentFinish
from agent_reason import react_agent_runnable , tools
# from react_state import AgentState

load_dotenv()

class AgentState(TypedDict):
  input : str
  agent_outcome : Union[AgentAction, AgentFinish , None]
  intermediate_steps : Annotated[list[tuple[AgentAction , str]] , operator.add]

def reason_node(state : AgentState):
  agent_outcome = react_agent_runnable.invoke(state)
  return {"agent_outcome" : agent_outcome}

def act_node(state : AgentState):
  agent_action = state["agent_outcome"]

  tool_name = agent_action.tool
  tool_input = agent_action.tool_input

  tool_function = None
  for tool in tools:
    if tool.name == tool_name:
      tool_function = tool
      break

  if tool_function:
    try:
      if isinstance(tool_input, dict):
        output = tool_function.invoke(tool_input)
      else:
        output = tool_function.invoke(tool_input)
    except Exception as e:
      output = f"Error executing tool '{tool_name}': {str(e)}"
  else:
    output = f"Tool '{tool_name}' not found"
  
  return {"intermediate_steps" : [(agent_action , str(output))]}




