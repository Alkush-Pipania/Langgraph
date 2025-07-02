from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

actor_prompt_template = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """You are expert AI researcher.
      Current time : {time}
      1. {first_instruction}
      2. Reflect and critique your answer. Be severe to maximize improvement.
      3. After the reflection, **list 1-3 search queries seperately** for researching improvements. Do not include them inside the reflection.
      """,
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the requirement format.")
  ]
).partial(
  time=lambda:datetime.datetime.now().isoformat(),
)

first_responseder_prompt_template = actor_prompt_template.partial(
  first_instruction="Answer the question in ~250 words, detailed and comprehensive.",
)

first_response_chain = first_responseder_prompt_template | llm.bind_tools(tools=[AnswerQuestion] , tool_choice="AnswerQuestion") 


