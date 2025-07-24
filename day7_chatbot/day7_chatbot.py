import argparse
import os
import openai
from dotenv import load_dotenv
import tiktoken
import streamlit as st


class ChatBot:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.max_token = 12000
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
        self.messages = [{'role': 'system', 'content':'你是一个能说会道，幽默风趣的陪聊。'}]
    def update_msg(self, msg):
        if self.count_tokens() >= self.max_token:
            for i, m in enumerate(self.messages):
                if m['role'] != 'system':
                    self.messages.pop(i)
                    break
        self.messages.append(msg)

    def update_user_msg(self, user_msg):
        msg = {'role': 'user', 'content': user_msg}
        self.update_msg(msg)
    def update_assistant_msg(self, assistant_msg):
        msg = {'role': 'assistant', 'content': assistant_msg}
        self.update_msg(msg)
    def count_tokens(self):
        encoding = tiktoken.encoding_for_model(self.model)

        # 每条 message 的固定开销（OpenAI 规则）
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for msg in self.messages:
            num_tokens += tokens_per_message
            for key, value in msg.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def talk(self, msg):
        self.update_user_msg(msg)
        rsp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.messages,
        )
        reply = rsp.choices[0].message.content
        self.update_assistant_msg(reply)
        return reply

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
parser.add_argument('--temperature', type=float, default=0.7)
args = parser.parse_args()
model, temperature = args.model, args.temperature
bot = ChatBot(model, temperature)

st.set_page_config(page_title="ChatBot", page_icon="💬")
# 初始化对话上下文
st.title("💬 小猪聊天机器人")

# 展示历史对话
for msg in bot.messages:
    if msg['role'] == 'system':
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("你想问什么？")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            reply = bot.talk(user_input)
            st.markdown(reply)


