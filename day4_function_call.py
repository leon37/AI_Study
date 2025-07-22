import argparse
import json
import os

import openai
from dotenv import load_dotenv

def get_nickname(name):
    record = {
        '胥邈': '小猪',
        '卷卷': '小卷崽',
    }
    return record.get(name, f"我不知道{name}的外号是什么")

class FuncCallClient:
    def __init__(self, model_name, temperature):
        load_dotenv()
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
        self.tools = [{
            "type": "function",
            "function": {
                "name": "get_nickname",
                "description": "当用户询问关于外号或别名相关的问题时，应该调用此方法",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "需要找到对应外号的名字"
                        }
                    }
                }
            }

        }]
        self.messages = []
    def send(self, msg):
        self.update_msg({"role": "user", "content": msg})
        rsp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto"
        )
        return rsp

    def send_func_msg(self, func_name, content, tool_call_id):
        msg = {"role": "tool", "name": func_name, "content": content, "tool_call_id": tool_call_id}
        self.update_msg(msg)
        rsp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=self.messages,
        )
        return rsp

    def update_msg(self, msg):
        self.messages.append(msg)

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="moonshot-v1-8k")
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()
    client = FuncCallClient(args.model, args.temperature)
    while True:
        msg = input()
        if msg == 'exit':
            break
        rsp = client.send(msg)
        message = rsp.choices[0].message
        client.update_msg(message)
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            func_name=tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if func_name == 'get_nickname':
                content = get_nickname(args.get("name"))
                final_rsp = client.send_func_msg(func_name, content, tool_call.id)
                print(final_rsp.choices[0].message.content)
                client.update_msg(final_rsp.choices[0].message)


if __name__ == '__main__':
    test()
