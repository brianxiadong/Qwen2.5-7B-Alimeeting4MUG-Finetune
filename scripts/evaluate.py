#!/usr/bin/env python3
"""
使用训练好的模型批量评估验证集
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(base_model_path, adapter_path):
    """加载模型和 LoRA adapter"""
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True  # 4-bit 量化
    )
    
    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_title(model, tokenizer, meeting_content):
    """生成会议主题标题"""
    prompt = f"""你是一个专业的会议助手。请根据以下会议内容片段，生成一个简洁准确的主题标题。
会议内容：
{meeting_content}"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate(model, tokenizer, eval_file, num_samples=10):
    """评估模型"""
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"评估 {min(num_samples, len(data))} 个样本")
    print(f"{'='*60}\n")
    
    correct = 0
    for i, item in enumerate(data[:num_samples]):
        input_text = item['input']
        expected = item['output']
        
        generated = generate_title(model, tokenizer, input_text)
        
        # 简单匹配检查
        match = expected in generated or generated in expected
        if match:
            correct += 1
        
        print(f"[样本 {i+1}]")
        print(f"  期望: {expected}")
        print(f"  生成: {generated}")
        print(f"  {'✅ 匹配' if match else '❌ 不匹配'}\n")
    
    accuracy = correct / num_samples * 100
    print(f"{'='*60}")
    print(f"简单匹配准确率: {accuracy:.1f}% ({correct}/{num_samples})")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/data/Alimeeting4MUG/models/Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter", default="./outputs/qwen2.5-7b-mug-lora")
    parser.add_argument("--eval_file", default="./data/dev_alpaca.json")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.base_model, args.adapter)
    evaluate(model, tokenizer, args.eval_file, args.num_samples)
