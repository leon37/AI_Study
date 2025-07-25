from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-3.5-turbo')
parser = StrOutputParser()
prompt_template = ChatPromptTemplate.from_messages([("user", "{user_input}")])
chain = prompt_template | model | parser

while True:
    user_input=input("user: ")
    if user_input == "exit":
        break

    result = chain.invoke(user_input)
    print(f"assistant: {result}")