# AML Terraformer

基于 LLM 的 AML 交易图扰动工具。使用 LLM 代理智能地对洗钱集群进行扰动，支持注入中间节点、合并账户和拆分账户三种操作。

## 快速开始

### 一键运行（推荐）

```bash
# 设置环境变量
export AML_DATA_DIR=/path/to/your/data
export DEEPSEEK_API_KEY="your-api-key"

# 运行
python scripts/run_perturbation.py \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --output output \
    --provider deepseek \
    --api-key $DEEPSEEK_API_KEY \
    --max-cluster-size 100
```

### 参数说明

- `--accounts`: 账户CSV文件路径（必需）
- `--transactions`: 交易CSV文件路径（必需）
- `--output`: 输出目录（必需）
- `--provider`: LLM提供商，可选 `deepseek` 或 `openai`（必需）
- `--api-key`: API密钥（必需）
- `--model`: 模型名称（可选，默认 deepseek-chat 或 gpt-4）
- `--max-steps`: 每个簇的最大扰动步数（可选，默认2）
- `--topk`: 候选数量（可选，默认10）

## 功能特性

### LLM 驱动的决策
使用 LLM 代理智能决定应用哪些扰动操作

### 三种扰动工具
- **inject_intermediary**: 在交易链中插入中间账户
- **merge_accounts**: 将多个账户合并为一个统一账户
- **split_account**: 将一个账户拆分为两个，重新分配交易

### 严格验证
所有操作都经过验证，失败时自动回滚

### 全面日志记录
详细的 JSONL 日志和摘要报告

## 输出文件

处理完成后，在输出目录会生成：

**最终结果**:
- `transactions_perturbed.csv` - 扰动后的所有交易
- `accounts_perturbed.csv` - 扰动后的所有账户
- `summary_report.json` - 汇总统计报告
- `perturb_log.jsonl` - 详细操作日志
- `final_comparison.png` - 最终对比可视化图

## 工作流程

```
1. 读取数据
   ↓
2. 数据预处理和聚类检测
   ↓
3. 对每个洗钱簇：
   a. LLM决策扰动操作
   b. 执行扰动（inject/merge/split）
   c. 验证结果
   d. 保存中间结果
   ↓
4. 保存最终结果和生成报告
```

## 可视化说明

### 节点颜色
- **青色** (#4ecdc4): 原始账户
- **红色** (#ff6b6b): 新创建的账户

### 边颜色
- **红色粗线** (#e74c3c): 洗钱交易
- **灰色细线** (#95a5a6): 普通交易

### 节点标签
- `Bank XX` - 银行ID
- `SPLIT` - 拆分创建的账户
- `INTERM` - 注入的中间账户

## 使用 CLI 包

如果需要更多控制，也可以直接使用 `aml_terraformer` 包的CLI：

```bash
python -m aml_terraformer.cli \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --outdir output \
    --llm-provider deepseek \
    --llm-api-key $DEEPSEEK_API_KEY \
    --max-steps-per-cluster 2
```

## 安装

### 基础安装

```bash
pip install -e .
```

### 安装额外依赖

```bash
# API 提供商
pip install -e ".[openai]"      # OpenAI
pip install -e ".[anthropic]"    # Anthropic Claude
pip install -e ".[deepseek]"     # DeepSeek

# 本地模型支持（Qwen 等）
pip install -e ".[local]"

# GBT 监控训练
pip install -e ".[gbt]"

# 安装所有依赖
pip install -e ".[all]"
```

### 环境变量配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入你的配置
```

## 项目结构

```
aml-terraformer/
├── README.md                    # 项目文档
├── LICENSE                      # MIT 许可证
├── .env.example                 # 环境变量模板
├── pyproject.toml               # 项目配置
├── setup.py                     # 安装脚本
├── data/                        # 数据文件（用户准备）
├── rule/                        # 规则配置
│   ├── data/rule.json          # 规则定义
│   └── README.md
├── scripts/                     # 脚本文件
│   ├── run_perturbation.py      # 一键运行脚本
│   ├── train_grpo.py           # GRPO 训练
│   ├── train_gbt_monitor.py    # GBT 监控训练
│   ├── evaluate_detection_rate.py # 检测率评估
│   └── analyze_rule_hits.py    # 规则命中分析
├── src/aml_terraformer/         # 主包
│   ├── agent/                  # LLM 代理
│   ├── core/                   # 核心功能
│   ├── io/                     # 数据读写
│   ├── monitor/                # 检测监控器
│   ├── pipeline/               # 处理流水线
│   ├── rl/                     # 强化学习
│   └── tools/                  # 扰动工具
├── evaluation/                  # 评估工具
└── docs/                       # 文档
```

## 注意事项

1. **大数据集**: 如果数据集很大（百万级交易），建议先用小样本测试
2. **API费用**: 每个簇会调用2-4次LLM API
3. **内存使用**: 处理大图时可能需要较多内存
4. **实时输出**: 所有处理进度都会实时打印到控制台

## 技术支持

问题反馈：查看 `perturb_log.jsonl` 了解详细的操作日志
