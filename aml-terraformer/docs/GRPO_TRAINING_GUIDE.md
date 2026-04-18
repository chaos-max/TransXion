# GRPO Training Guide

本指南介绍如何使用 GRPO (Group Relative Policy Optimization) 训练本地模型。

## 两种训练模式

### 1. 数据收集模式 (`--train-mode=collect`)

**功能**: 收集轨迹数据和计算优势值，保存为 JSONL 文件，但**不更新模型参数**。

**适用场景**:
- 使用 API 模型（DeepSeek/OpenAI）
- 为后续的 API fine-tuning 收集数据
- 离线强化学习数据收集

**示例**:
```bash
# 使用 DeepSeek API 收集数据
python scripts/train_grpo.py \
    --accounts data/account-sample.csv \
    --transactions data/trans-sample.csv \
    --output output/grpo_collect \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --num-clusters 5 \
    --num-samples 4 \
    --train-mode collect

# 使用本地模型收集数据（不训练）
python scripts/train_grpo.py \
    --accounts data/account-sample.csv \
    --transactions data/trans-sample.csv \
    --output output/grpo_collect \
    --provider local \
    --model /path/to/Qwen3-1.7B \
    --device cuda \
    --num-clusters 5 \
    --num-samples 4 \
    --train-mode collect
```

**输出**:
- `training_examples_final.jsonl`: 训练样本数据
- `cluster_statistics.json`: 每个 cluster 的统计信息
- `overall_statistics.json`: 总体训练统计

---

### 2. 训练模式 (`--train-mode=train`)

**功能**: 使用 GRPO 算法进行**真正的梯度更新**，训练本地模型参数。

**适用场景**:
- 使用本地 transformer 模型（如 Qwen3-1.7B）
- 需要直接更新模型参数
- 在线强化学习训练

**重要**: 此模式**仅支持 `--provider=local`**

**示例**:
```bash
# 基础训练（全参数微调）
python scripts/train_grpo.py \
    --accounts data/account-sample.csv \
    --transactions data/trans-sample.csv \
    --output output/grpo_trained \
    --provider local \
    --model /path/to/Qwen3-1.7B \
    --device cuda \
    --num-clusters 5 \
    --num-samples 4 \
    --train-mode train \
    --learning-rate 1e-5 \
    --batch-size 1 \
    --gradient-accumulation-steps 4

# 使用 LoRA 进行高效微调
python scripts/train_grpo.py \
    --accounts data/account-sample.csv \
    --transactions data/trans-sample.csv \
    --output output/grpo_lora \
    --provider local \
    --model /path/to/Qwen3-1.7B \
    --device cuda \
    --num-clusters 10 \
    --num-samples 4 \
    --train-mode train \
    --learning-rate 1e-4 \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --batch-size 2
```

**输出**:
- `checkpoint-{N}/`: 定期保存的模型 checkpoint
- `checkpoint-final/`: 最终训练完成的模型
- `training_stats.json`: 训练统计（包括 policy loss）

---

## 参数说明

### 奖励权重参数

这些参数控制 GRPO 训练的奖励函数：

1. **`--w-detection`** (默认: 10.0) - **检测奖励权重**
   - 控制"降低检测概率"这个目标的重要性
   - 计算: `detection_reward = (score_before - score_after) × w_detection`
   - 权重越大，模型越重视降低被监控系统检测到的概率

2. **`--w-validity`** (默认: 1.0) - **有效性奖励权重**
   - 奖励执行有效操作
   - 操作成功时给予 `+w_validity` 奖励，失败时给予 `-5.0` 惩罚

3. **`--w-budget`** (默认: 0.1) - **预算惩罚权重**
   - 控制操作成本的惩罚力度（当前版本未使用）

4. **`--w-final-bonus`** (默认: 20.0) - **最终成功奖励权重**
   - 当智能体成功完成任务时的额外奖励
   - 条件: 主动停止且最终检测分数 < 0.3 阈值

**总奖励公式**:
```
total_reward = validity_reward + detection_reward + final_bonus
             = w_validity + (Δdetection × w_detection) + w_final_bonus
```

### 训练超参数（仅 train 模式）

- **`--learning-rate`** (默认: 1e-5)
  - 学习率，建议范围: 1e-6 到 1e-4
  - LoRA 训练可以使用较大的学习率（如 1e-4）

- **`--num-epochs`** (默认: 1)
  - 每个 cluster 的训练轮数
  - 增加此值可以更充分地利用收集的数据

- **`--batch-size`** (默认: 1)
  - 每个 mini-batch 的样本数
  - GPU 显存允许时可以增大

- **`--gradient-accumulation-steps`** (默认: 4)
  - 梯度累积步数
  - 实际 batch size = batch_size × gradient_accumulation_steps

- **`--max-grad-norm`** (默认: 1.0)
  - 梯度裁剪的最大范数
  - 防止梯度爆炸

### LoRA 参数

- **`--use-lora`**
  - 启用 LoRA 高效微调
  - 需要安装: `pip install peft`

- **`--lora-r`** (默认: 8)
  - LoRA 秩（rank）
  - 越大模型容量越大，但参数也越多

- **`--lora-alpha`** (默认: 16)
  - LoRA 缩放参数
  - 通常设置为 lora_r 的 2 倍

---

## GRPO 算法原理

GRPO (Group Relative Policy Optimization) 是一种策略梯度算法：

1. **采样轨迹**: 为每个 cluster 采样 K 个轨迹（trajectories）

2. **计算优势值**: 使用组内相对优势
   ```
   advantage_i = (return_i - mean(returns)) / std(returns)
   ```

3. **策略梯度更新**:
   ```
   Loss = -E[advantage × log π(action|state)]
   ```

4. **梯度下降**: 更新模型参数以最大化加权 log 概率

**关键特性**:
- 组内归一化避免了全局 baseline 的需求
- 适合小样本在线学习
- 相对优势减少方差

---

## 常见问题

### Q1: 训练模式和数据收集模式的区别？

**数据收集模式**:
- ✅ 收集轨迹数据
- ✅ 计算优势值
- ✅ 保存 JSONL 文件
- ❌ **不更新**模型参数

**训练模式**:
- ✅ 收集轨迹数据
- ✅ 计算优势值
- ✅ 计算 policy gradient loss
- ✅ **更新**模型参数（反向传播）
- ✅ 保存 checkpoint

### Q2: 为什么之前的脚本说 "Use data for fine-tuning"？

因为之前只实现了数据收集模式（`GRPOTrainer`），需要：
1. 运行脚本收集数据
2. 手动使用 OpenAI/DeepSeek fine-tuning API 训练

现在新增的训练模式（`GRPOLocalTrainer`）可以直接训练本地模型，无需两步操作。

### Q3: 什么时候用 LoRA？

**推荐使用 LoRA**:
- ✅ GPU 显存有限
- ✅ 需要快速实验
- ✅ 模型较大（> 1B 参数）

**全参数微调**:
- ✅ GPU 显存充足
- ✅ 有大量训练数据
- ✅ 追求最佳性能

### Q4: 如何继续训练已有的 checkpoint？

目前脚本不支持从 checkpoint 恢复训练。如需实现：
1. 修改 `train_grpo.py` 添加 `--resume-from` 参数
2. 在创建 client 时加载保存的 checkpoint

### Q5: 训练需要多少数据？

建议：
- **最少**: 5-10 个 clusters，每个 4 个样本 (20-40 trajectories)
- **推荐**: 20-50 个 clusters，每个 4-8 个样本 (80-400 trajectories)
- **充分训练**: 100+ clusters

---

## 依赖安装

### 基础依赖
```bash
pip install torch transformers
```

### LoRA 支持
```bash
pip install peft
```

### 量化加载（可选）
```bash
pip install bitsandbytes  # 8-bit/4-bit 量化
```

---

## 示例工作流

### 场景 1: 快速实验（LoRA）

```bash
# 1. 用小数据集快速测试
python scripts/train_grpo.py \
    --accounts data/account-sample.csv \
    --transactions data/trans-sample.csv \
    --output output/test_run \
    --provider local \
    --model /path/to/Qwen3-1.7B \
    --device cuda \
    --num-clusters 2 \
    --num-samples 2 \
    --train-mode train \
    --use-lora

# 2. 正式训练
python scripts/train_grpo.py \
    --accounts data/account-reference.csv \
    --transactions data/trans-reference.csv \
    --output output/grpo_lora_trained \
    --provider local \
    --model /path/to/Qwen3-1.7B \
    --device cuda \
    --num-clusters 20 \
    --num-samples 4 \
    --train-mode train \
    --use-lora \
    --learning-rate 1e-4 \
    --batch-size 2
```

### 场景 2: API 数据收集 + 后续 Fine-tuning

```bash
# 1. 使用 DeepSeek API 收集高质量数据
python scripts/train_grpo.py \
    --accounts data/account-reference.csv \
    --transactions data/trans-reference.csv \
    --output output/grpo_api_data \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --num-clusters 50 \
    --num-samples 8 \
    --train-mode collect

# 2. 使用 DeepSeek Fine-tuning API
# (需要手动上传 training_examples_final.jsonl)

# 或者用收集的数据训练本地模型
# (需要编写自定义训练脚本加载 JSONL 数据)
```

---

## 参考资料

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [Qwen Models](https://github.com/QwenLM/Qwen) - Qwen 模型文档

---

## 更新日志

### v2.0 (当前版本)
- ✅ 新增 `GRPOLocalTrainer` 支持本地模型训练
- ✅ 新增 `--train-mode` 参数选择模式
- ✅ 支持 LoRA 高效微调
- ✅ 支持梯度累积和裁剪
- ✅ 自动保存训练 checkpoint

### v1.0 (原版本)
- ✅ 数据收集模式（`GRPOTrainer`）
- ✅ 支持 API 和本地模型推理
- ✅ GRPO 优势计算
