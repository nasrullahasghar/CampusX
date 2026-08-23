from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()

url = "https://jang.com.pk/en/71530-daily-sugary-drinks-linked-to-25-times-higher-stomach-cancer-risk-study-news"

loader = WebBaseLoader(
    web_path=url
)
docs = loader.load()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Answer the following question \n {question} from the following text.\n{text}",
    input_variables=["question","text"]
)

chain = prompt | model | parser

result = chain.invoke({"question":"What is the summary of the text","text":docs[0].page_content})
print(result)