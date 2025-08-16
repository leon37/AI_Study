from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
lora_adapter = 'chradden/TinyLlama-1.1B-Chat-v1.0-bf16-lora-adapter'

tokenizer = AutoTokenizer.from_pretrained(base_model)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map='auto',
    quantization_config=bnb_config)

model = PeftModel.from_pretrained(model, lora_adapter)
model.eval()

text = 'Explain the difference between LoRA and QLoRA in one sentence.'
inputs = tokenizer(text, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))