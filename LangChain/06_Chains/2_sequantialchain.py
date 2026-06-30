from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Give me detailed note on the {topic}",
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary on the {text}",
    input_variables = ['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"Unemployment in Pakistan"})
print(result)

chain.get_graph().print_ascii()