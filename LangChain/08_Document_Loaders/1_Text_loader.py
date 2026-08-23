import os
os.environ['TRANSFORMERS_VERBOSITY'] = "error"
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
prompt = PromptTemplate(
    template = "Write a Summary of the following poem - \n {poem}",
    input_variables=["poem"]
)
parser = StrOutputParser()
loader = TextLoader("cricket.txt",encoding="utf-8")

docs = loader.load()

print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | llm | parser

summary = chain.invoke({"poem":docs[0].page_content})
print(f"The Summary of the Poem is:\n{summary}")