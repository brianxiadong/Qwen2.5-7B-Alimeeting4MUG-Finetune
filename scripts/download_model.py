#!/usr/bin/env python3
"""
使用 ModelScope 下载 Qwen2.5-7B 模型

ModelScope 在国内访问速度更快，推荐在国内服务器使用
"""

import argparse
import os
from pathlib import Path


def download_model(model_id: str, cache_dir: str, revision: str = "master"):
    """
    从 ModelScope 下载模型
    
    Args:
        model_id: ModelScope 模型ID
        cache_dir: 本地缓存目录
        revision: 模型版本
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("请先安装 modelscope: pip install modelscope")
        return None
    
    print(f"开始下载模型: {model_id}")
    print(f"保存目录: {cache_dir}")
    
    model_dir = snapshot_download(
        model_id=model_id,
        cache_dir=cache_dir,
        revision=revision
    )
    
    print(f"\n✅ 模型下载完成!")
    print(f"模型路径: {model_dir}")
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载 Qwen2.5 模型")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen2.5-7B",
        help="ModelScope 模型ID (默认: Qwen/Qwen2.5-7B)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./models",
        help="模型保存目录 (默认: ./models)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="模型版本 (默认: master)"
    )
    
    args = parser.parse_args()
    
    # 创建缓存目录
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    model_dir = download_model(args.model_id, args.cache_dir, args.revision)
    
    if model_dir:
        print(f"\n📝 使用提示:")
        print(f"在 configs/train_lora.yaml 中修改:")
        print(f"  model_name_or_path: {model_dir}")


if __name__ == "__main__":
    main()
