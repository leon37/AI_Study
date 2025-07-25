from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-3.5-turbo')
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{profession}专家，请用通俗易懂的语言回答用户的问题。"),
    ("user", "{question}")
])

chain = prompt | model | parser

try:
    while True:
        profession = input("请输入职业: ")
        question = input("请输入问题: ")
        result = chain.invoke({"profession": profession, "question": question})
        print(f"assistant: {result}")
except Exception as e:
    print(e)