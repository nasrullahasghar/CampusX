from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
You are a research assistant.

Summarize the research paper:
{paper_input}

Explanation Style:
{style_input}

Explanation Length:
{length_input}


Requirements:

1. Mathematical Details:
- Include important equations if available.
- Explain mathematical concepts simply.
- Add small code examples where useful.

2. Analogies:
- Use simple real-world analogies.

If information is unavailable, reply:
"Insufficient information available"

Make the explanation accurate and clear.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)

template.save("prompt.json")