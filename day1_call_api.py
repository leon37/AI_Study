import openai

if __name__ == "__main__":
    api_key = 'sk-s5iDj0PxpZZnAV0yMCcttlxkZWzm6AHxNimIchCAiCYoOUe4'
    api_base = 'https://api.moonshot.cn/v1'
    messages = [
        {"role": "system", "content": "你是一位性格毒舌、说话直白的程序员助理。"},
        {"role": "user", "content": "Python 和 Go 哪个更好"},
    ]
    ai = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = ai.chat.completions.create(
        model='moonshot-v1-8k',
        temperature=0.7,
        messages=messages,
    )
    print(response.choices[0].message.content)