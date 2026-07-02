from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Write a detailed report on the {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Provide a concise executive summary in one paragraph.{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

def is_long_text(text: str) -> bool:
    return len(text.split(" ")) > 300


branch_chain = RunnableBranch(
    (is_long_text, {"text": RunnablePassthrough()} | prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({"topic": "cricket"})
print("\n--- Final Output ---")
print(result)
