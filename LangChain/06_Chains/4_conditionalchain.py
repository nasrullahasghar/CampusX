from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch , RunnableParallel , RunnableLambda
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
)

class Feedback(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Give sentiment of the coustomer either it's positive or negative.")

pydantic_parser = PydanticOutputParser(name="sentiment of the feedback",pydantic_object=Feedback)
text_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Classify the feedback either it's positive or negative \n{feedback}\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction":pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template = "Give me appropriate response of this positive feedback \n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Give me appropriate response of this negative feedback \n{feedback}",
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | text_parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | text_parser),
    RunnableLambda(lambda x : "Could Not Find Sentiment")
)


chain = classifier_chain | branch_chain

feedback = "Hii, How are you?"

result = chain.invoke({"feedback":feedback})

print(result)

# chain.get_graph().print_ascii()