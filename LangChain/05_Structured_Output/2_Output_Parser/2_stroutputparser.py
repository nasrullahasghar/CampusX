from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# Report
template1 = PromptTemplate(
    template="Write a detailed report on the following {topic}",
    input_variables=['topic']
)
# Summary
template2 = PromptTemplate(
    template="Write a 5 line summaryon the following  text.\n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Islamabad"})

print(result)