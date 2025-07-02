from typing_extensions import Annotated , TypedDict
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

class Joke(TypedDict):
  """Joke to tell user."""

  setup: Annotated[str, ... , "The setup of the joke"]

  punchline : Annotated[str , ... , "The punchline of the joke"]
  rating : Annotated[Optional[int], None , "How funny the joke is , from 1 to 10"]

structure_llm = llm.with_structured_output(Joke)

response = structure_llm.invoke("Tell me a joke about cats sitting on table")

print(response)