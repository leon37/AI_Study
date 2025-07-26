from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
prompt = ChatPromptTemplate([
    ("system", "你是一个很有帮助的助手，尽你的能力来回答所有问题。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])
model = ChatOpenAI(model='gpt-3.5-turbo')
chain = prompt | model

demo_history = ChatMessageHistory()
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_history,
    input_message_key="input",
    history_key="chat_history",
)

try:
    print("你好，我是你的专属AI，有什么可以帮助你的？")
    while True:
        user_input = input("user:")
        if user_input == "exit":
            print("Goodbye")
            break
        ret = chain_with_message_history.invoke(
            {"input": user_input},
            {"configurable": {"session_id": "unused"}}
        ).get("content")
        print(f"assistant: {ret}")
except KeyboardInterrupt:
    print("Goodbye")