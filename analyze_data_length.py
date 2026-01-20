#!/usr/bin/env python3
"""
分析 AliMeeting4MUG 训练数据的 token 长度分布
"""

import json
import os
from collections import Counter

# 尝试导入 matplotlib，如果没有就用文本输出
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将只输出文本统计")

# 尝试导入 tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("警告: transformers 未安装，将使用字符长度估算")

def analyze_data(data_path, model_path=None):
    """分析数据长度分布"""
    
    # 加载数据
    print(f"正在加载数据: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 计算长度
    lengths = []
    
    if HAS_TOKENIZER and model_path and os.path.exists(model_path):
        print(f"使用 tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        for i, item in enumerate(data):
            # 构建完整文本 (instruction + input + output)
            text = ""
            if "instruction" in item:
                text += item["instruction"]
            if "input" in item:
                text += item["input"]
            if "output" in item:
                text += item["output"]
            
            tokens = tokenizer.encode(text)
            lengths.append(len(tokens))
            
            if (i + 1) % 500 == 0:
                print(f"已处理 {i+1}/{len(data)} 样本...")
    else:
        print("使用字符长度估算 (约 1.5 字符 = 1 token)")
        for item in data:
            text = ""
            if "instruction" in item:
                text += item["instruction"]
            if "input" in item:
                text += item["input"]
            if "output" in item:
                text += item["output"]
            # 中文大约 1.5 字符 = 1 token
            lengths.append(int(len(text) / 1.5))
    
    return lengths

def print_statistics(lengths):
    """打印统计信息"""
    lengths_sorted = sorted(lengths)
    n = len(lengths)
    
    print("\n" + "="*60)
    print("📊 Token 长度统计")
    print("="*60)
    print(f"  样本总数:     {n}")
    print(f"  最小长度:     {min(lengths)}")
    print(f"  最大长度:     {max(lengths)}")
    print(f"  平均长度:     {sum(lengths)/n:.0f}")
    print(f"  中位数:       {lengths_sorted[n//2]}")
    print(f"  75分位数:     {lengths_sorted[int(n*0.75)]}")
    print(f"  90分位数:     {lengths_sorted[int(n*0.90)]}")
    print(f"  95分位数:     {lengths_sorted[int(n*0.95)]}")
    print(f"  99分位数:     {lengths_sorted[int(n*0.99)]}")
    
    print("\n" + "="*60)
    print("📏 cutoff_len 覆盖率分析")
    print("="*60)
    
    thresholds = [1024, 2048, 4096, 6144, 8192, 16384]
    for t in thresholds:
        count = sum(1 for l in lengths if l <= t)
        pct = count / n * 100
        truncated = sum(1 for l in lengths if l > t)
        status = "✅" if pct >= 95 else "⚠️" if pct >= 80 else "❌"
        print(f"  cutoff_len={t:5d}: {status} 覆盖 {pct:5.1f}% ({count}/{n}), 截断 {truncated} 样本")
    
    print("\n" + "="*60)
    print("📈 长度分布直方图 (文本版)")
    print("="*60)
    
    # 创建区间
    bins = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), 
            (4096, 8192), (8192, 16384), (16384, float('inf'))]
    
    for low, high in bins:
        count = sum(1 for l in lengths if low <= l < high)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        label = f"{low}-{high}" if high != float('inf') else f"{low}+"
        print(f"  {label:12s}: {bar:25s} {count:4d} ({pct:4.1f}%)")
    
    return lengths_sorted

def plot_distribution(lengths, output_path="token_length_distribution.png"):
    """绘制长度分布图"""
    if not HAS_MATPLOTLIB:
        print("\n无法绘制图表: matplotlib 未安装")
        print("安装命令: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 直方图
    ax1 = axes[0]
    ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=2048, color='orange', linestyle='--', linewidth=2, label='cutoff=2048')
    ax1.axvline(x=4096, color='red', linestyle='--', linewidth=2, label='cutoff=4096')
    ax1.axvline(x=8192, color='purple', linestyle='--', linewidth=2, label='cutoff=8192')
    ax1.set_xlabel('Token Length', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Token Length Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 累积分布
    ax2 = axes[1]
    sorted_lengths = sorted(lengths)
    cumulative = [i/len(lengths)*100 for i in range(1, len(lengths)+1)]
    ax2.plot(sorted_lengths, cumulative, color='steelblue', linewidth=2)
    ax2.axhline(y=95, color='green', linestyle='--', linewidth=1, label='95% coverage')
    ax2.axvline(x=2048, color='orange', linestyle='--', linewidth=2, label='cutoff=2048')
    ax2.axvline(x=4096, color='red', linestyle='--', linewidth=2, label='cutoff=4096')
    ax2.axvline(x=8192, color='purple', linestyle='--', linewidth=2, label='cutoff=8192')
    ax2.set_xlabel('Token Length', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 分布图已保存: {output_path}")

def main():
    # 配置路径 - 根据实际情况修改
    data_path = "./data/train_alpaca.json"
    model_path = "/data/Alimeeting4MUG/models/Qwen/Qwen2.5-7B"
    output_img = "./token_length_distribution.png"
    
    # 检查数据文件
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        print("请确保在正确的目录下运行此脚本")
        return
    
    # 分析数据
    lengths = analyze_data(data_path, model_path)
    
    # 打印统计
    print_statistics(lengths)
    
    # 绘制图表
    plot_distribution(lengths, output_img)
    
    print("\n" + "="*60)
    print("💡 建议")
    print("="*60)
    sorted_lengths = sorted(lengths)
    p95 = sorted_lengths[int(len(lengths)*0.95)]
    p99 = sorted_lengths[int(len(lengths)*0.99)]
    
    if p95 <= 2048:
        print("  推荐 cutoff_len: 2048 (覆盖 95%+ 数据)")
    elif p95 <= 4096:
        print("  推荐 cutoff_len: 4096 (覆盖 95%+ 数据)")
    elif p95 <= 8192:
        print("  推荐 cutoff_len: 8192 (覆盖 95%+ 数据)")
    else:
        print(f"  数据较长，95分位数={p95}，建议考虑截断或分段处理")
    
    print(f"  (当前 95分位数: {p95}, 99分位数: {p99})")

if __name__ == "__main__":
    main()
