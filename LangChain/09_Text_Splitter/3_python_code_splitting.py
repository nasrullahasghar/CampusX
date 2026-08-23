from langchain_text_splitters import PythonCodeTextSplitter
# we csn use other language textslitters also
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""


splitter2 = PythonCodeTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

chunks2 = splitter2.split_text(text)

print(len(chunks2))
print(chunks2)