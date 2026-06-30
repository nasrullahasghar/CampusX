from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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

prompt1 = template1.invoke({"topic":"Dusra Delivery in Cricket"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

final_result = model.invoke(prompt2)

print(final_result.content)