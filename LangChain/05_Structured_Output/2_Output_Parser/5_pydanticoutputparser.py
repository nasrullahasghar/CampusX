from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field , StrictInt


load_dotenv()


model =  ChatGroq(model = "llama-3.3-70b-versatile")

# Pydantic Schema
class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: StrictInt = Field(gt = 18 ,description="Age of the Person")
    city: str = Field(description="Name of the City the Person Belongs to")
    

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template = "Give me the name , age and city of a fictional {place} person\n  {format_instruction}",
    input_variables=["place"],
    partial_variables = {"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place":"pakistani"})

print(result)