from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = "Explain the following joke - {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1 , model , parser, prompt2 , model ,  parser)

response = chain.invoke({"topic":"AI"})
print(response)