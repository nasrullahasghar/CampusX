import os
import certifi
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

chat_history = [
    SystemMessage(content="You are a helpful AI assistant!")
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content)

print(chat_history)