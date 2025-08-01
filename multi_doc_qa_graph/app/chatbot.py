from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langgraph.types import Command

from multi_doc_qa_graph.app import State

def init_prompt(context, user_input):
    system_prompt = (
        "你是一个问答助手，以下是相关背景资料，请根据背景资料回答用户问题，"
        "如果找不到答案，就说不知道。\n\n背景资料：\n"
        f"{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"{user_input}"),
        ]
    )
    return prompt

def chatbot(state: State):
    client = ChatOpenAI(model='gpt-4o-mini')
    question = state.get('question', "")
    if len(question) == 0:
        raise ValueError('question is empty')
    prompt = init_prompt(state.get('retrieved_chunks', ''), question)
    messages = prompt.format_messages()
    ai_message = client.invoke(messages)

    print("Assistant: ", ai_message.content)
    return Command(update={'messages': state.get('messages', [])+messages+[ai_message]})
