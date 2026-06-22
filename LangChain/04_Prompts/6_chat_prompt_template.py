# For Dynamic Messages
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    # In ChatPromptTemplate we use only tuple and write role and message in it.
    ('system', 'You are a helpful {domain} expert'), 
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)