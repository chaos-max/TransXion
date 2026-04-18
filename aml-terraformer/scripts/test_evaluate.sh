#!/bin/bash

# 简单测试脚本 - 仅评估 LoRA 模型（跳过基础模型以加快测试）
# 使用 1 个集群和 2 个样本进行快速测试
#
# 使用前设置环境变量:
#   export AML_DATA_DIR=/path/to/your/data
#   export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B
#   export AML_ROOT=/path/to/aml-terraformer

echo "开始快速评估测试..."
echo "注意: 这只是一个测试，使用 1 个集群和 2 个样本"
echo ""

# 设置默认值（如果未设置）
AML_ROOT=${AML_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
AML_DATA_DIR=${AML_DATA_DIR:-data}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/path/to/Qwen2.5-7B}

cd "$AML_ROOT"

python scripts/evaluate_detection_rate.py \
  --accounts "$AML_DATA_DIR/accounts.csv" \
  --transactions "$AML_DATA_DIR/transactions.csv" \
  --base-model "$QWEN_MODEL_PATH" \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation_test \
  --num-clusters 1 \
  --num-samples 2 \
  --max-steps 2 \
  --skip-base \
  --device cuda

echo ""
echo "测试完成！检查是否有错误。"
