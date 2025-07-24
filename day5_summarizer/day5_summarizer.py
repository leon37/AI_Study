import os

import openai
import argparse
from dotenv import load_dotenv

def summerize(text):
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
    messages = [{"role":"system", "content":"你是一个擅长总结处理中英文文章内容的助手。"},
                {"role":"user", "content":f"请将下面这段文字总结压缩为50字到100字的摘要。\n\n{text}"},]
    resp = client.chat.completions.create(
        model='moonshot-v1-8k',
        temperature=0.3,
        messages=messages,
    )
    return resp.choices[0].message.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    args = parser.parse_args()
    filepath = args.filepath
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    ret = summerize(content)
    print(ret)