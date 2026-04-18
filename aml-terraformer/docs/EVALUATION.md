# 检测率评估文档

本目录包含用于评估和对比基础模型与 LoRA 训练模型的检测逃避性能的脚本。

## 概述

评估衡量模型修改交易数据以逃避基于规则的 AML 检测的能力，同时保持数据有效性。提供了两个评估脚本：

1. **quick_evaluate_detection.py** - 快速分析训练结果
2. **evaluate_detection_rate.py** - 基础模型与 LoRA 模型的完整对比

## 快速开始

### 1. 分析训练结果

从检查点查看训练性能指标：

```bash
python scripts/quick_evaluate_detection.py \
  --training-output output/test_train_lora_rulebased/grpo_train/checkpoint-375
```

**输出内容:**
- 训练期间获得的平均回报（奖励）
- 训练迭代的改进情况
- 最佳/最差性能

**结果解读:**
- 正回报 = 成功降低检测分数
- 更高的回报 = 更好的逃避能力
- 随时间改进 = 模型正在学习

### 2. 完整模型对比

在测试集群上对比基础模型与 LoRA 训练模型：

```bash
# 完整评估（推荐使用 16GB+ 显存的 GPU）
# 设置环境变量
export AML_DATA_DIR=/path/to/your/data
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B

python scripts/evaluate_detection_rate.py \
  --accounts $AML_DATA_DIR/accounts.csv \
  --transactions $AML_DATA_DIR/transactions.csv \
  --base-model $QWEN_MODEL_PATH \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation \
  --num-clusters 10 \
  --num-samples 3 \
  --device cuda

# 快速测试（3个集群，适合测试）
python scripts/evaluate_detection_rate.py \
  --accounts $AML_DATA_DIR/accounts.csv \
  --transactions $AML_DATA_DIR/transactions.csv \
  --base-model $QWEN_MODEL_PATH \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation_quick \
  --num-clusters 3 \
  --num-samples 2 \
  --device cuda

# 仅评估 LoRA 模型（跳过基础模型以加快速度）
python scripts/evaluate_detection_rate.py \
  --accounts $AML_DATA_DIR/accounts.csv \
  --transactions $AML_DATA_DIR/transactions.csv \
  --base-model $QWEN_MODEL_PATH \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation_lora_only \
  --num-clusters 5 \
  --skip-base \
  --device cuda
```

## 理解结果

### 训练结果 (quick_evaluate_detection.py)

```
训练性能:
  Mean return: 2.0848 ± 1.3992
  Best return: 5.1000

训练改进:
  Improvement: +0.2098
```

**结果解读:**
- **平均回报 = 2.08**: 平均而言，模型获得正回报，意味着成功降低了检测分数
- **改进 = +0.21**: 模型在训练期间有所改进，表明它从经验中学习
- **最佳回报 = 5.10**: 模型在某些集群上可以获得高回报，显示出强大的逃避能力

### 完整评估结果 (evaluate_detection_rate.py)

完整评估的输出示例：

```
基础模型:
  平均初始检测分数: 0.8500
  平均最终检测分数: 0.7200
  平均分数降低:     0.1300 (15.29%)

LoRA 模型（训练后）:
  平均初始检测分数: 0.8500
  平均最终检测分数: 0.6100
  平均分数降低:     0.2400 (28.24%)

改进（LoRA vs 基础）:
  最终分数改进:     +0.1100 (越低越好)
  分数降低改进:     +0.1100

✓ LoRA 模型实现了更低的检测率（更好的逃避能力）
```

**结果解读:**
- **初始分数相同**: 两个模型从相同的集群开始
- **LoRA 实现更低的最终分数**: 训练后的模型在逃避检测方面更好
- **正向改进**: LoRA 相比基础模型额外降低了 11% 的检测率

## 关键指标

1. **检测分数**: 集群被标记为洗钱的概率 (0-1)
   - 越高 = 越可疑
   - 目标: 降低此分数

2. **分数降低**: 初始分数 - 最终分数
   - 正值 = 成功逃避
   - 越高 = 性能越好

3. **相对降低**: (初始 - 最终) / 初始
   - 检测分数的百分比降低

4. **成功率**: 成功逃避的集群百分比

## 输出文件

完整评估将详细结果保存到 JSON:

```
output/evaluation/
  evaluation_results.json  # 包含每个集群详细信息的完整结果
```

JSON 结构:
```json
{
  "timestamp": "2026-01-08T...",
  "config": {...},
  "results": {
    "base": {
      "avg_initial_score": 0.85,
      "avg_final_score": 0.72,
      "avg_score_reduction": 0.13,
      "cluster_results": [...]
    },
    "lora": {
      "avg_initial_score": 0.85,
      "avg_final_score": 0.61,
      "avg_score_reduction": 0.24,
      "cluster_results": [...]
    }
  },
  "comparison": {
    "score_reduction_improvement": 0.11,
    "is_better": true
  }
}
```

## 性能考虑

### 内存需求

- **仅基础模型**: ~14GB 显存 (Qwen2.5-7B FP16 格式)
- **两个模型**: 需要顺序评估（运行之间卸载模型）
- **8位量化**: 使用 `--load-in-8bit` 降至 ~7GB 显存
- **CPU 模式**: 使用 `--device cpu`（速度慢得多）

### 速度

- **每个集群评估**: ~2-5 分钟（取决于 max_steps 和 num_samples）
- **10个集群，3个样本**: 两个模型约 ~60-150 分钟
- **快速测试（3个集群）**: ~20-30 分钟

### 优化技巧

1. **从小规模开始**: 首先使用 `--num-clusters 3` 进行验证
2. **跳过基础模型**: 如果只关心 LoRA 性能，使用 `--skip-base`
3. **减少样本**: 使用 `--num-samples 2` 而不是 3
4. **使用量化**: 添加 `--load-in-8bit` 以降低显存使用

## 故障排除

### CUDA 内存不足

```bash
# 方案 1: 使用 8 位量化
python scripts/evaluate_detection_rate.py ... --load-in-8bit

# 方案 2: 使用 CPU
python scripts/evaluate_detection_rate.py ... --device cpu

# 方案 3: 分别评估模型
python scripts/evaluate_detection_rate.py ... --skip-lora  # 仅基础模型
python scripts/evaluate_detection_rate.py ... --skip-base  # 仅 LoRA 模型
```

### 模型加载错误

确保路径正确：
- 基础模型: `$QWEN_MODEL_PATH`
- LoRA 检查点: `output/test_train_lora_rulebased/grpo_train/checkpoint-XXX`

### 性能慢

- 减少 `--num-clusters` 和 `--num-samples`
- 减少 `--max-steps`（默认: 2）
- 使用 GPU 而不是 CPU

## 示例工作流程

```bash
# 1. 快速检查训练结果
python scripts/quick_evaluate_detection.py \
  --training-output output/test_train_lora_rulebased/grpo_train/checkpoint-375

# 2. 小规模测试（3个集群）
python scripts/evaluate_detection_rate.py \
  --accounts $AML_DATA_DIR/accounts.csv \
  --transactions $AML_DATA_DIR/transactions.csv \
  --base-model $QWEN_MODEL_PATH \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation_test \
  --num-clusters 3 \
  --num-samples 2 \
  --device cuda

# 3. 如果测试效果好，运行完整评估
python scripts/evaluate_detection_rate.py \
  --accounts $AML_DATA_DIR/accounts.csv \
  --transactions $AML_DATA_DIR/transactions.csv \
  --base-model $QWEN_MODEL_PATH \
  --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-375 \
  --output output/evaluation_full \
  --num-clusters 10 \
  --num-samples 3 \
  --device cuda

# 4. 分析结果
cat output/evaluation_full/evaluation_results.json
```

## 对比多个检查点

比较不同的训练检查点：

```bash
# 检查点 100
python scripts/evaluate_detection_rate.py ... \
  --lora-checkpoint output/.../checkpoint-100 \
  --output output/eval_ckpt100

# 检查点 200
python scripts/evaluate_detection_rate.py ... \
  --lora-checkpoint output/.../checkpoint-200 \
  --output output/eval_ckpt200

# 检查点 375（最终）
python scripts/evaluate_detection_rate.py ... \
  --lora-checkpoint output/.../checkpoint-375 \
  --output output/eval_ckpt375

# 对比结果
python -c "
import json
for ckpt in [100, 200, 375]:
    with open(f'output/eval_ckpt{ckpt}/evaluation_results.json') as f:
        data = json.load(f)
        lora = data['results']['lora']
        print(f'检查点 {ckpt}: {lora[\"avg_score_reduction\"]:.4f}')
"
```
