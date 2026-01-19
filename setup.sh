#!/bin/bash
# Qwen2.5-7B AliMeeting4MUG LoRA 微调环境配置脚本
# 用法: bash setup.sh

set -e

echo "========================================"
echo "Qwen2.5-7B AliMeeting4MUG 微调环境配置"
echo "========================================"

# 尝试初始化 conda
init_conda() {
    # 常见的 conda 安装路径
    CONDA_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "/opt/conda"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/work/anaconda3"
        "/work/miniconda3"
        "/data/anaconda3"
        "/data/miniconda3"
    )
    
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
            echo "🔍 找到 conda: $conda_path"
            source "$conda_path/etc/profile.d/conda.sh"
            return 0
        fi
    done
    
    # 如果上面都没找到，尝试使用 which conda
    if which conda &> /dev/null; then
        CONDA_BIN=$(which conda)
        CONDA_BASE=$(dirname $(dirname $CONDA_BIN))
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            echo "🔍 找到 conda: $CONDA_BASE"
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    
    return 1
}

# 检查 conda 是否可用
if ! init_conda; then
    echo "❌ 未找到 conda，请先安装 Miniconda 或 Anaconda"
    echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "   或者手动初始化 conda:"
    echo "   source /path/to/conda/etc/profile.d/conda.sh"
    exit 1
fi

# 环境名称
ENV_NAME="qwen_finetune"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 ${ENV_NAME} 已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境..."
        conda activate ${ENV_NAME}
    fi
fi

# 创建 conda 环境
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 创建 Conda 环境: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# 激活环境
conda activate ${ENV_NAME}

echo "✅ 环境已激活: ${ENV_NAME}"

# 检测 CUDA 版本
echo "🔍 检测 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    echo "   检测到 CUDA: ${CUDA_VERSION}"
else
    echo "   未检测到 nvcc，使用默认 CUDA 11.8"
    CUDA_VERSION="11.8"
fi

# 安装 PyTorch
echo "📦 安装 PyTorch..."
if [[ "$CUDA_VERSION" == "12."* ]]; then
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# 安装项目依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 克隆 LLaMA-Factory (如果不存在)
if [ ! -d "LLaMA-Factory" ]; then
    echo "📦 克隆 LLaMA-Factory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

# 安装 LLaMA-Factory
echo "📦 安装 LLaMA-Factory..."
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# 尝试安装 Flash Attention (可能失败)
echo "📦 尝试安装 Flash Attention 2 (可选)..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention 安装失败，跳过"

# 验证安装
echo ""
echo "========================================"
echo "验证安装"
echo "========================================"
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✅ CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✅ GPU 数量: {torch.cuda.device_count()}')" 2>/dev/null || true
llamafactory-cli version

echo ""
echo "========================================"
echo "✅ 环境配置完成!"
echo "========================================"
echo ""
echo "下一步操作:"
echo "  1. 激活环境: conda activate ${ENV_NAME}"
echo "  2. 下载模型: python scripts/download_model.py"
echo "  3. 预处理数据: python scripts/preprocess_data.py"
echo "  4. 开始训练: llamafactory-cli train configs/train_lora.yaml"
echo ""
