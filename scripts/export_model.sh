#!/bin/bash
# 导出合并后的模型脚本

echo "========================================="
echo "🔄 导出合并后的模型"
echo "========================================="

# 检查是否在正确的目录
if [ ! -f "configs/export.yaml" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 LoRA adapter 是否存在
if [ ! -d "outputs/qwen2.5-7b-mug-lora" ]; then
    echo "❌ LoRA adapter 不存在: outputs/qwen2.5-7b-mug-lora"
    exit 1
fi

# 检查基础模型是否存在
if [ ! -d "models/Qwen/Qwen2.5-7B" ]; then
    echo "❌ 基础模型不存在: models/Qwen/Qwen2.5-7B"
    exit 1
fi

echo "📦 开始导出..."
echo "   基础模型: ./models/Qwen/Qwen2.5-7B"
echo "   LoRA adapter: ./outputs/qwen2.5-7b-mug-lora"
echo "   输出目录: ./outputs/qwen2.5-7b-mug-merged"
echo ""

llamafactory-cli export configs/export.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 导出成功!"
    echo "========================================="
    echo "合并后的模型位于: ./outputs/qwen2.5-7b-mug-merged"
    echo ""
    echo "📝 启动 vLLM 服务:"
    echo "   vllm serve ./outputs/qwen2.5-7b-mug-merged --trust-remote-code --port 30000 --max-model-len 4096"
else
    echo ""
    echo "❌ 导出失败"
    exit 1
fi
