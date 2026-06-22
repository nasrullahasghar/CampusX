import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Fix SSL certificate issue
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,load_prompt


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


st.header("📄 Research Paper Summarizer")


paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)



template = load_prompt("prompt.json")



if st.button("Summarize"):
    chain = template | llm

    with st.spinner("Generating summary..."):

        response = chain.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })

    st.write(response.content)