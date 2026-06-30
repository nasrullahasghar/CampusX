# It is  just a topic, and  StructuredOutputParser is removed by LangChain officially
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# Schema
schema = [
    ResponseSchema(name = "fact1" , description = "Fact 1 about the topic"),
    ResponseSchema(name = "fact2" , description = "Fact 2 about the topic"),
    ResponseSchema(name = "fact3" , description = "Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template  = PromptTemplate(
    template = "Give three facts about the {topic},{format_instruction}",
    input_variables=['topic'],
    partial_variables = {"format_instruction":parser.get_format_instruction}
)

chain = template | model | parser

result = chain.invoke({"topic":"Pakistan"})

print(result)



