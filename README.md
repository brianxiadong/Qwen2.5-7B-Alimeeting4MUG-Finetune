# Qwen2.5-7B AliMeeting4MUG LoRA Fine-tuning

使用 **LLaMA-Factory** 对 Qwen2.5-7B 进行 LoRA 微调，训练模型执行会议理解与生成（MUG）任务。

## � 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/brianxiadong/Qwen2.5-7B-Alimeeting4MUG-Finetune.git
cd Qwen2.5-7B-Alimeeting4MUG-Finetune

# 2. 一键配置环境 (自动创建 conda 环境并安装所有依赖)
bash setup.sh

# 3. 激活环境
conda activate qwen_finetune

# 4. 下载模型
python scripts/download_model.py

# 5. 预处理数据
python scripts/preprocess_data.py

# 6. 开始训练
llamafactory-cli train configs/train_lora.yaml
```

## �📋 目录

- [项目简介](#项目简介)
- [数据集说明](#数据集说明)
- [环境配置](#环境配置)
- [模型下载](#模型下载)
- [数据预处理](#数据预处理)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [常见问题](#常见问题)

---

## 项目简介

本项目基于阿里巴巴 **AliMeeting4MUG** 数据集，使用 LoRA（Low-Rank Adaptation）技术对 Qwen2.5-7B 大语言模型进行高效微调。

### 支持的 MUG 任务

| 任务 | 英文 | 说明 |
|------|------|------|
| 主题标题生成 | Topic Title Generation (TTG) | 为会议片段生成简洁的主题标题 |
| 抽取式摘要 | Extractive Summarization (ES) | 从会议中提取关键句子作为摘要 |
| 主题分割 | Topic Segmentation (TS) | 识别会议中的主题边界 |
| 关键词提取 | Keyphrase Extraction (KPE) | 提取会议关键词 |
| 行动项检测 | Action Item Detection (AID) | 检测会议中的待办事项 |

---

## 数据集说明

### 概述

AliMeeting4MUG 是阿里巴巴发布的大规模中文会议理解语料库，包含 654 场录制的普通话会议，每场会议 15-30 分钟，涉及 2-4 名参与者。

### 文件结构

```
dataset/
├── train.csv    # 训练集 (296 条会议, ~30MB)
└── dev.csv      # 验证集 (66 条会议, ~7MB)
```

### CSV 格式

每个 CSV 文件包含两列：

| 列名 | 说明 |
|------|------|
| `idx` | 样本索引 (0, 1, 2, ...) |
| `content` | JSON 格式的会议数据 |

### Content JSON 结构

```json
{
  "meeting_key": "M0138",
  
  "topic_segment_ids": [
    {
      "id": 88,
      "candidate": [
        {
          "title": "文艺晚会找领导讲话并安排座位",
          "key_sentence": ["6", "24", "45"]
        },
        {
          "title": "如何安排文艺晚会的座位",
          "key_sentence": ["60", "77"]
        }
      ]
    }
  ],
  
  "sentence_list": [
    {
      "id": 1,
      "speaker": "no.0",
      "start_time": "0.0",
      "end_time": "5.2",
      "s": "今天我们来讨论一下晚会的安排。"
    },
    {
      "id": 2,
      "speaker": "no.1", 
      "start_time": "5.5",
      "end_time": "10.1",
      "s": "好的，我们先从座位开始。"
    }
  ],
  
  "paragraph_segment_ids": [
    {"id": 3}, {"id": 10}, {"id": 25}
  ],
  
  "action_ids": [
    {"id": 45}, {"id": 120}
  ]
}
```

### 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `meeting_key` | string | 会议唯一标识符 |
| `topic_segment_ids` | array | 主题分段信息，每个分段包含 ID 和候选标题 |
| `topic_segment_ids[].id` | int | 该主题段结束的句子 ID |
| `topic_segment_ids[].candidate` | array | 候选标题列表（通常 3 个） |
| `candidate[].title` | string | 主题标题 |
| `candidate[].key_sentence` | array | 该主题的关键句子 ID 列表 |
| `sentence_list` | array | 完整会议转录 |
| `sentence_list[].id` | int | 句子 ID |
| `sentence_list[].speaker` | string | 说话人标识 (no.0, no.1, ...) |
| `sentence_list[].start_time` | string | 开始时间（秒） |
| `sentence_list[].end_time` | string | 结束时间（秒） |
| `sentence_list[].s` | string | 句子文本内容 |
| `paragraph_segment_ids` | array | 段落分段点的句子 ID |
| `action_ids` | array | 行动项句子的 ID |

---

## 环境配置

### 1. 创建 Conda 环境

```bash
# 创建新的 Python 3.10 环境
conda create -n qwen_finetune python=3.10 -y
conda activate qwen_finetune

# 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
# pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 克隆项目和 LLaMA-Factory

```bash
# 克隆本项目
git clone https://github.com/brianxiadong/Qwen2.5-7B-Alimeeting4MUG-Finetune.git
cd Qwen2.5-7B-Alimeeting4MUG-Finetune

# 克隆 LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 Flash Attention 2（推荐）

Flash Attention 2 是一种高效的注意力机制实现，可以：
- ⚡ **加速训练** 1.5-2 倍
- 💾 **减少显存占用** 5-20 倍（针对注意力层）
- 📈 **支持更长序列** 而不会 OOM

#### 方式一：使用预编译 wheel（推荐）

由于 Flash Attention 编译很慢，建议直接下载预编译的 wheel 文件：

```bash
# 1. 检测你的环境版本
python scripts/check_flash_attn_env.py

# 或使用一行命令快速检测
python -c "import torch; import sys; v=sys.version_info; print(f'Python: cp{v.major}{v.minor}, PyTorch: {torch.__version__.split(\"+\")[0]}, CUDA: {torch.version.cuda}, CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

输出示例：
```
Python: cp312, PyTorch: 2.5.0, CUDA: 12.1, CXX11_ABI: False
```

2. 根据输出，到 [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) 下载对应版本：

| 环境 | wheel 文件名 |
|------|-------------|
| Python 3.12 + PyTorch 2.5 + CUDA 12 + ABI=False | `flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl` |
| Python 3.12 + PyTorch 2.5 + CUDA 12 + ABI=True | `flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` |
| Python 3.10 + PyTorch 2.1 + CUDA 11.8 | `flash_attn-2.8.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` |

3. 下载并安装：
```bash
# 下载 (替换为你的版本)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-xxx.whl

# 安装
pip install flash_attn-xxx.whl
```

#### 方式二：从源码编译（慢，约 10-30 分钟）

```bash
pip install flash-attn --no-build-isolation
```

> ⚠️ 编译需要大量 RAM（建议 32GB+）和 CUDA 开发环境。

#### 验证安装

```bash
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} installed successfully!')"
```

### 5. 安装 DeepSpeed（多 GPU 训练）

```bash
pip install deepspeed
```

### 5. 验证安装

```bash
# 验证 LLaMA-Factory
llamafactory-cli version

# 验证 PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 模型下载

### 使用 ModelScope 下载（推荐国内用户）

```bash
# 安装 modelscope
pip install modelscope

# 下载 Qwen2.5-7B 模型
python scripts/download_model.py --model_id Qwen/Qwen2.5-7B --cache_dir ./models
```

下载完成后，修改 `configs/train_lora.yaml` 中的模型路径：
```yaml
model_name_or_path: ./models/Qwen/Qwen2.5-7B
```

### 其他可选模型

| 模型 | ModelScope ID | 显存需求 |
|------|---------------|----------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | ~24GB (LoRA) |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~24GB (LoRA) |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | ~12GB (LoRA) |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | ~8GB (LoRA) |

---

## 数据预处理

### 运行预处理脚本

```bash
# 进入项目目录
cd /path/to/Qwen2.5-7B-Alimeeting4MUG-Finetune

# 执行数据转换（默认：主题标题生成任务）
python scripts/preprocess_data.py

# 或指定其他任务
python scripts/preprocess_data.py --task extractive_summary
```

### 检查输出

```bash
# 查看生成的训练数据
head -n 3 data/train_alpaca.json
```

预期输出格式：
```json
[
  {
    "instruction": "你是一个专业的会议助手。请根据以下会议内容片段，生成一个简洁准确的主题标题。",
    "input": "会议内容：\n[no.0]: 今天我们来讨论一下晚会的安排。\n[no.1]: 好的，我们先从座位开始。",
    "output": "文艺晚会找领导讲话并安排座位"
  }
]
```

### 复制数据集配置

**重要**：需要将 `dataset_info.json` 复制到 LLaMA-Factory 的 data 目录，或将生成的数据文件复制过去：

```bash
# 方式1：复制配置到 LLaMA-Factory
cp data/dataset_info.json /path/to/LLaMA-Factory/data/
cp data/*.json /path/to/LLaMA-Factory/data/

# 方式2：在配置中使用绝对路径
# 修改 configs/train_lora.yaml 中的 dataset_dir 为绝对路径
```

---

## 模型训练

### 基础训练命令

```bash
cd /path/to/LLaMA-Factory

# 使用项目配置文件训练
llamafactory-cli train /path/to/Qwen2.5-7B-Alimeeting4MUG-Finetune/configs/train_lora.yaml
```

### 显存不足时使用量化

编辑 `configs/train_lora.yaml`，取消注释量化配置：

```yaml
# 4-bit 量化 (适用于 16GB 显存 GPU)
quantization_bit: 4
quantization_method: bitsandbytes
```

### 多 GPU 训练

```bash
# 使用 DeepSpeed ZeRO-2
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/train_lora.yaml \
    --deepspeed examples/deepspeed/ds_z2_config.json
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 64 | LoRA 秩，越大表达能力越强 |
| `lora_alpha` | 128 | LoRA 缩放因子 |
| `learning_rate` | 2e-4 | 学习率 |
| `num_train_epochs` | 3 | 训练轮数 |
| `per_device_train_batch_size` | 2 | 每 GPU 批次大小 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `cutoff_len` | 2048 | 最大序列长度 |

---

## 模型推理

### 交互式对话

```bash
cd /path/to/LLaMA-Factory

llamafactory-cli chat /path/to/Qwen2.5-7B-Alimeeting4MUG-Finetune/configs/inference.yaml
```

### 示例对话

```
User: 请根据以下会议内容生成主题标题：
[no.0]: 我们来讨论一下下周的产品发布会。
[no.1]: 发布会的场地已经确定了吗？
[no.0]: 确定了，在公司大会议室。
[no.1]: 好的，那我们需要准备哪些材料？

Assistant: 产品发布会场地及材料准备讨论
```

### 合并 LoRA 权重到基础模型

```bash
llamafactory-cli export configs/merge.yaml
```

---

## 模型验证与评估

### 训练损失 (Loss) 参考标准

| 阶段 | Loss 范围 | 说明 |
|------|-----------|------|
| 初始 | 2.0 - 5.0 | 训练刚开始的损失 |
| 收敛后 | 0.5 - 1.5 | 较好的收敛状态 |
| 理想目标 | 0.3 - 0.8 | 会议生成任务的合理范围 |

> **注意**: Loss 过低（< 0.1）可能意味着过拟合，需检查验证集 loss 是否同步下降。

### 推荐 GPU 配置

| GPU | 显存 | 推荐配置 | 预估训练时间 |
|-----|------|----------|--------------|
| **A800 80GB** | 80GB | batch_size=8, cutoff_len=4096, lora_rank=128 | ~25 分钟 |
| **A100 80GB** | 80GB | batch_size=8, cutoff_len=4096, lora_rank=128 | ~25 分钟 |
| **A100 40GB** | 40GB | batch_size=2, cutoff_len=2048, lora_rank=64 | ~40 分钟 |
| **RTX 4090** | 24GB | batch_size=1, cutoff_len=2048, 4bit量化 | ~60 分钟 |

### 验证训练效果

#### 1. 查看训练日志

```bash
# 查看训练状态
cat outputs/qwen2.5-7b-mug-lora/trainer_state.json | python -m json.tool
```

#### 2. 运行评估脚本

```bash
# 在验证集上评估模型
python scripts/evaluate.py \
    --model_path outputs/qwen2.5-7b-mug-lora \
    --data_path data/dev_alpaca.json \
    --output_path outputs/eval_results.json
```

#### 3. 评估指标

| 指标 | 说明 | 良好范围 |
|------|------|----------|
| ROUGE-L | 生成文本与参考的最长公共子序列 | > 0.4 |
| BLEU-4 | N-gram 匹配精度 | > 0.3 |
| Exact Match | 完全匹配率 | > 0.1 |

### 常见训练问题排查

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| Loss 不下降 | 训练多 epoch 后 loss 仍 > 3.0 | 检查数据格式、增大学习率 |
| Loss 过高 | 最终 loss > 2.0 | 增加 epoch、增大 batch_size |
| 过拟合 | 训练 loss 降但验证 loss 升 | 增加 dropout、减少 epoch |

---

## 常见问题

### Q1: 显存不足 (OOM)

**解决方案：**
1. 启用 4-bit 量化：在 `train_lora.yaml` 中取消注释 `quantization_bit: 4`
2. 减小批次大小：`per_device_train_batch_size: 1`
3. 减小序列长度：`cutoff_len: 1024`
4. 使用梯度检查点：`gradient_checkpointing: true`

### Q2: 训练速度慢

**解决方案：**
1. 安装 Flash Attention 2：`pip install flash-attn --no-build-isolation`
2. 启用 bf16 训练（需要 Ampere 及以上 GPU）
3. 使用多 GPU 训练

### Q3: 模型输出质量差

**解决方案：**
1. 增加训练轮数
2. 调整 LoRA rank（尝试 128 或 256）
3. 检查数据质量，确保预处理正确

### Q4: 如何使用 Web UI 训练？

```bash
cd LLaMA-Factory
llamafactory-cli webui
```

然后在浏览器中打开 http://localhost:7860

---

## 项目结构

```
Qwen2.5-7B-Alimeeting4MUG-Finetune/
├── dataset/                    # 原始数据集
│   ├── train.csv
│   └── dev.csv
├── data/                       # 处理后的数据
│   ├── dataset_info.json       # LLaMA-Factory 数据集配置
│   ├── train_alpaca.json       # 训练数据 (Alpaca 格式)
│   └── dev_alpaca.json         # 验证数据 (Alpaca 格式)
├── configs/                    # 配置文件
│   ├── train_lora.yaml         # 训练配置
│   └── inference.yaml          # 推理配置
├── scripts/                    # 脚本
│   └── preprocess_data.py      # 数据预处理
├── outputs/                    # 训练输出 (自动生成)
│   └── qwen2.5-7b-mug-lora/    # LoRA 权重
└── README.md                   # 项目文档
```

---

## 参考资料

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 技术报告](https://qwenlm.github.io/blog/qwen2.5/)
- [AliMeeting4MUG 论文](https://arxiv.org/abs/2302.08466)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

---

## License

MIT License
