# ===== file: data/build_sample_dataset.py =====
import json
from pathlib import Path

SAMPLE = [
    {
        "user": "请用三句话介绍Python的优势。",
        "assistant": (
            "Python语法简洁、生态丰富、跨平台能力强。"
            "它拥有大量第三方库，适用于数据科学、Web与自动化。"
            "社区活跃，学习门槛低，入门与进阶都很友好。"
        ),
    },
    {
        "user": "写一段50字以内的早安问候。",
        "assistant": (
            "早安！愿你今天精神饱满，心情明朗，工作顺利，遇见美好的人和事。"
            "记得吃早餐，多喝水，保持微笑～"
        ),
    },
    {
        "user": "给我一个SQL示例：查询最近7天下单且金额超过100的用户数。",
        "assistant": (
            "SELECT COUNT(DISTINCT user_id) AS users FROM orders "
            "WHERE order_amount > 100 AND order_time >= NOW() - INTERVAL 7 DAY;"
        ),
    },
]

OUT = Path(__file__).resolve().parent / "sample_sft.jsonl"

# 方案A：极简伪模板（和样例一致）
# <|user|>...<|assistant|>...

def to_pseudo_chat(user: str, assistant: str) -> str:
    return f"<|user|>{user}<|assistant|>{assistant}"

# 方案B（TODO）：调用Qwen聊天模板（需在训练脚本里实现真正的 apply_chat_template）
# 这里先保留占位符，供你在脚本中选择使用。

def main():
    with OUT.open("w", encoding="utf-8") as f:
        for ex in SAMPLE:
            text = to_pseudo_chat(ex["user"], ex["assistant"])  # 或替换为真正模板
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print("Wrote:", OUT)

if __name__ == "__main__":
    main()