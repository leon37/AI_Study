import argparse
import openai

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="moonshot-v1-8k")
parser.add_argument("--temperature", type=float, default=0.5)
args = parser.parse_args()
api_key = 'sk-s5iDj0PxpZZnAV0yMCcttlxkZWzm6AHxNimIchCAiCYoOUe4'
api_base = 'https://api.moonshot.cn/v1'
messages = [
    {"role": "system", "content": "你是一位性格毒舌、说话直白的程序员助理。"},
    {"role": "user", "content": "Python 和 Go 哪个更好"},
]
ai = openai.OpenAI(api_key=api_key, base_url=api_base)
print(f"model: {args.model} temperature: {args.temperature}")
response = ai.chat.completions.create(
    model=args.model,
    temperature=args.temperature,
    messages=messages,
)
print(response.choices[0].message.content)
