from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def calculate(expression: str) -> str:
    """输入一个数学表达式字符串，返回计算结果。支持加减乘除和括号。"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"

def nickname(name: str) -> str:
    record = {
        '胥邈': '小猪',
        '李听': '卷卷'
    }
    return record.get(name, f"我不知道{name}的外号")

load_dotenv()
prompt = ChatPromptTemplate([
    ("system", "你是一个很有帮助的助手，尽你的能力来回答所有问题，并在需要的时候调用工具。"),
    ("human", "{input}")
])
model = ChatOpenAI(model='gpt-3.5-turbo')
tools = [calculate, nickname]
agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

try:
    print("你好，我是你的专属AI，有什么可以帮助你的？")
    while True:
        user_input = input("user:")
        if user_input == "exit":
            print("Goodbye")
            break
        ret = agent_executor.invoke(
            {"input": user_input},
        )

        print(f"assistant: {ret}")
except KeyboardInterrupt:
    print("Goodbye")