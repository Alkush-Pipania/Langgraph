from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

class Country(BaseModel):
     """Information about a country"""
     
     name: str = Field(description="name of the country")
     language: str = Field(description="language of the country")
     capital: str = Field(description="Capital of the country")


structured_llm = llm.with_structured_output(Country)


response = structured_llm.invoke("Tell me about France")

print(response)
# Output: name='France' language='French' capital='Paris'
