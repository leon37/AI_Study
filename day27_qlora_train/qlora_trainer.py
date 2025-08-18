# ===== file: qlora_trainer.py =====
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

BASE_ID = "Qwen/Qwen1.5-1.8B"
OUT_DIR = Path("outputs/qlora_trainer")
DATA_PATH = Path("data/sample_sft.jsonl")

assert torch.cuda.is_available(), "CUDA 不可用，请先正确安装含CUDA的PyTorch"

# ---------- tokenizer ----------

def build_tokenizer(base_id: str):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ---------- model (4bit) ----------

def build_4bit_base(base_id: str):
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=qconf,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return model

# ---------- lora ----------

def wrap_lora(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lconf = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lconf)
    return model

# ---------- dataset ----------

def load_sft_dataset(path: Path):
    ds = load_dataset("json", data_files=str(path))
    return ds["train"]

# 将 text -> token ids，构造成 causal LM 的 inputs/labels
# 关键点：把 padding 的 label 置为 -100，以免影响 loss

def tokenize_fn(examples: Dict[str, List[str]], tokenizer, max_len: int = 1024):
    texts = examples["text"]
    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,  # 由collator动态padding
        return_attention_mask=True,
    )
    out["labels"] = out["input_ids"].copy()
    return out

# 直接使用 DataCollatorForLanguageModeling(mlm=False) 也可以；
# 若需更细控制，可自写 collator，把 pad 部分的 labels 置为 -100。

def build_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------- trainer ----------

def build_trainer(model, tokenizer, train_ds):
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=build_collator(tokenizer),
    )
    return trainer

# ---------- main ----------

def main():
    tokenizer = build_tokenizer(BASE_ID)
    model = build_4bit_base(BASE_ID)
    model = wrap_lora(model)

    raw_train = load_sft_dataset(DATA_PATH)
    tokenized = raw_train.map(
        lambda x: tokenize_fn(x, tokenizer, max_len=1024),
        batched=True,
        remove_columns=raw_train.column_names,
    )

    trainer = build_trainer(model, tokenizer, tokenized)
    record = {}
    questions = [
        '请用三句话介绍Python的优势。',
        '写一段50字以内的早安问候。',
        '给我一个SQL示例：查询最近7天下单且金额超过100的用户数。'
    ]
    for question in questions:
        record.update({question: []})
    for q in record.keys():
        before_train_answer = model.generate(q)
        record[q].append(before_train_answer)
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR / "adapter"))
    for q in record.keys():
        after_train_answer = model.generate(q)
        record[q].append(after_train_answer)
    with open('compare.txt', 'w', encoding='utf-8') as f:
        for q, answers in record.items():
            before, after = answers[0], answers[1]
            f.write(f'question: {q}\n')
            f.write(f'before_train_answer: {before}')
            f.write(f'after_train_answer: {after}')

if __name__ == "__main__":
    main()