#!/usr/bin/env python3
"""
数据预处理脚本：将 AliMeeting4MUG CSV 格式转换为 LLaMA-Factory 兼容的 Alpaca 格式

AliMeeting4MUG 数据集支持多种 MUG (Meeting Understanding and Generation) 任务:
1. Topic Title Generation (TTG) - 主题标题生成
2. Extractive Summarization (ES) - 抽取式摘要
3. Topic Segmentation (TS) - 主题分割
4. Keyphrase Extraction (KPE) - 关键词提取
5. Action Item Detection (AID) - 行动项检测

本脚本默认处理 Topic Title Generation 任务
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# 增加 CSV 字段大小限制，以处理大型 JSON 内容
csv.field_size_limit(sys.maxsize)


def parse_meeting_content(content_str: str) -> Dict[str, Any]:
    """解析 CSV 中的 JSON 内容"""
    try:
        return json.loads(content_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return None


def extract_transcript(sentence_list: List[Dict]) -> str:
    """
    从 sentence_list 中提取会议转录文本
    格式: [说话人]: 内容
    """
    if not sentence_list:
        return ""
    
    transcript_lines = []
    for sent in sentence_list:
        speaker = sent.get("speaker", "unknown")
        text = sent.get("s", "")
        if text.strip():
            transcript_lines.append(f"[{speaker}]: {text}")
    
    return "\n".join(transcript_lines)


def extract_topic_segments(content: Dict) -> List[Dict[str, Any]]:
    """
    提取主题分段信息
    每个分段包含: segment_id, 候选标题列表, 对应的句子范围
    """
    topic_segments = content.get("topic_segment_ids", [])
    sentences = content.get("sentences", [])
    
    segments = []
    prev_end_id = 0
    
    for i, seg in enumerate(topic_segments):
        seg_id = seg.get("id", 0)
        candidates = seg.get("candidate", [])
        
        # 获取该分段对应的句子（从上一个分段结束到当前分段ID）
        segment_sentences = [s for s in sentences 
                           if prev_end_id <= s.get("id", 0) <= seg_id]
        
        if candidates and segment_sentences:
            # 使用第一个候选标题作为 ground truth
            title = candidates[0].get("title", "")
            transcript = extract_transcript(segment_sentences)
            
            segments.append({
                "segment_id": seg_id,
                "title": title,
                "transcript": transcript,
                "candidates": candidates
            })
        
        prev_end_id = seg_id + 1
    
    return segments


def create_title_generation_samples(content: Dict, meeting_key: str) -> List[Dict]:
    """
    为主题标题生成任务创建训练样本
    """
    samples = []
    segments = extract_topic_segments(content)
    
    for seg in segments:
        if seg["transcript"] and seg["title"]:
            sample = {
                "instruction": "你是一个专业的会议助手。请根据以下会议内容片段，生成一个简洁准确的主题标题。",
                "input": f"会议内容：\n{seg['transcript']}",
                "output": seg["title"]
            }
            samples.append(sample)
    
    return samples


def create_extractive_summary_samples(content: Dict, meeting_key: str) -> List[Dict]:
    """
    为抽取式摘要任务创建训练样本
    """
    samples = []
    segments = extract_topic_segments(content)
    
    for seg in segments:
        if not seg["transcript"] or not seg["candidates"]:
            continue
            
        # 从 key_sentence 获取关键句子ID
        candidates = seg["candidates"]
        if candidates:
            key_sentence_ids = candidates[0].get("key_sentence", [])
            sentences = content.get("sentences", [])
            
            # 提取关键句子
            key_sentences = []
            for sent in sentences:
                if str(sent.get("id")) in key_sentence_ids:
                    key_sentences.append(sent.get("s", ""))
            
            if key_sentences:
                sample = {
                    "instruction": "你是一个专业的会议助手。请从以下会议内容中提取最关键的句子作为摘要。",
                    "input": f"会议内容：\n{seg['transcript']}",
                    "output": "\n".join(key_sentences)
                }
                samples.append(sample)
    
    return samples


def process_csv_file(input_path: str, task: str = "title_generation") -> List[Dict]:
    """
    处理 CSV 文件，生成训练样本
    
    Args:
        input_path: 输入 CSV 文件路径
        task: 任务类型 (title_generation, extractive_summary)
    
    Returns:
        训练样本列表
    """
    all_samples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            content_str = row.get('content', '')
            content = parse_meeting_content(content_str)
            
            if content is None:
                continue
            
            meeting_key = content.get("meeting_key", "unknown")
            
            if task == "title_generation":
                samples = create_title_generation_samples(content, meeting_key)
            elif task == "extractive_summary":
                samples = create_extractive_summary_samples(content, meeting_key)
            else:
                print(f"未知任务类型: {task}")
                continue
            
            all_samples.extend(samples)
    
    return all_samples


def save_alpaca_format(samples: List[Dict], output_path: str):
    """保存为 Alpaca 格式的 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(samples)} 条样本到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AliMeeting4MUG 数据预处理")
    parser.add_argument("--train_input", type=str, default="dataset/train.csv",
                        help="训练集 CSV 文件路径")
    parser.add_argument("--dev_input", type=str, default="dataset/dev.csv",
                        help="验证集 CSV 文件路径")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="输出目录")
    parser.add_argument("--task", type=str, default="title_generation",
                        choices=["title_generation", "extractive_summary"],
                        help="任务类型")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理训练集
    if Path(args.train_input).exists():
        print(f"处理训练集: {args.train_input}")
        train_samples = process_csv_file(args.train_input, args.task)
        save_alpaca_format(train_samples, output_dir / "train_alpaca.json")
    else:
        print(f"训练集文件不存在: {args.train_input}")
    
    # 处理验证集
    if Path(args.dev_input).exists():
        print(f"处理验证集: {args.dev_input}")
        dev_samples = process_csv_file(args.dev_input, args.task)
        save_alpaca_format(dev_samples, output_dir / "dev_alpaca.json")
    else:
        print(f"验证集文件不存在: {args.dev_input}")


if __name__ == "__main__":
    main()
