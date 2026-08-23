# It is most widely used text splitter
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = """
Senior researcher Dr. Andrew Chan stated, “This is the first study to demonstrate an association between sugar-sweetened beverage intake and gastric cancer in a U.S. population.

Gastric cancer is the fifth-leading cause of cancer death worldwide, but we’ve known very little about how diet might contribute to this cancer, Chan added.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)