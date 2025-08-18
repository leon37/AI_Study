# ===== file: qlora_sft_trainer.py =====
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer

BASE_ID = "Qwen/Qwen1.5-1.8B"
OUT_DIR = Path("outputs/qlora_sft")
DATA_PATH = Path("data/sample_sft.jsonl")

# ========== Step 0. 环境 & 断言 ==========
assert torch.cuda.is_available(), "CUDA 不可用，请先正确安装含CUDA的PyTorch"

# ========== Step 1. 构建分词器 ==========

def build_tokenizer(base_id: str):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    # TODO: 若 tok.pad_token 为空，为其指定一个（常用 eos_token）
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ========== Step 2. 4bit量化加载基座 ==========

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

# ========== Step 3. 选取 LoRA 目标模块（Qwen/LLaMA系常见命名） ==========

def target_modules_for_qwen():
    # TODO: 如需更严谨，打印 model.named_modules() 过滤包含 q/k/v/o 的线性层名
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# ========== Step 4. 注入 LoRA ==========

def wrap_lora(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lconf = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules_for_qwen(),
    )
    model = get_peft_model(model, lconf)
    return model

# ========== Step 5. 数据集加载 ==========

def load_sft_dataset(path: Path):
    # 期望JSONL里有字段 `text`
    ds = load_dataset("json", data_files=str(path))
    # ds["train"][0]["text"] 应该是一条完整的指令-回复拼接文本
    return ds["train"]

# ========== Step 6. 构建 Trainer ==========

def build_trainer(model, tokenizer, train_ds):
    # TODO: 如果你想用真正的 Qwen chat 模板，可在此对 train_ds.map 应用 apply_chat_template 生成 text
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
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=1024,  # TODO: 2060 吃紧可降到 512/768
        packing=False,
        args=args,
    )
    return trainer

# ========== Step 7. 训练 & 保存 ==========

def main():
    tokenizer = build_tokenizer(BASE_ID)
    model = build_4bit_base(BASE_ID)
    model = wrap_lora(model)

    print("Trainable parameters:")
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    train_ds = load_sft_dataset(DATA_PATH)

    trainer = build_trainer(model, tokenizer, train_ds)
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR / "adapter"))  # 保存LoRA适配器

    # TODO: 训练前后对比一次generate，保存到 OUT_DIR / "compare.txt"

if __name__ == "__main__":
    main()