import os
import certifi
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SSL_CERT_FILE"] = certifi.where()
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , SystemMessage,AIMessage
load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

messages = [
    SystemMessage(content = "You are a Helpful Assistent"),
    HumanMessage(content = "Tell me about LangChain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

