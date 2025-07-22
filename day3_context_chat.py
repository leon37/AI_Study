import os

import openai
import argparse
from dotenv import load_dotenv

class AIClient:
    def __init__(self, model, temperature):
        load_dotenv()
        self.openai = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'))
        self.messages = [{"role": "system", "content": "你是一位性格毒舌、说话直白的程序员助理。"}]
        self.model = model
        self.temperature = temperature
    def update(self, msg, role):
        send_msg = {"role": role, "content": msg}
        self.messages.append(send_msg)
    def talk(self, msg):
        self.update(msg, "user")
        rsp = self.openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.messages,
        )
        content = rsp.choices[0].message.content
        self.update(content, "assistant")
        return content


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="moonshot-v1-8k")
parser.add_argument("--temperature", type=float, default=0.5)
args = parser.parse_args()
client = AIClient(args.model, args.temperature)
while True:
    userContent = input()
    if userContent == "exit":
        break
    response = client.talk(userContent)
    print(response)