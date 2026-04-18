# 规则引擎（Rule-based AML Detection Pipeline）使用说明

## 1. 项目简介

本项目实现了一套**基于规则的反洗钱（AML）检测流水线**，支持从原始交易数据出发，自动完成：

1. 规则特征构建（Rule Feature Engineering）
2. 基于语义规则配置的规则命中判断（Rule Engine）
3. 一键脚本运行完整流程（Pipeline Automation）

系统采用**规则语义层与数据物理层解耦**的设计，具备良好的可维护性与可扩展性。

------

## 2. 项目目录结构

```
rule/
├── data/                          # 规则配置
│   └── rule.json                  # 规则配置文件（语义层）
│
├── out/                           # 输出结果目录（自动生成）
│
├── build_rule_features.py         # 规则特征构建脚本
├── run_rule_engine.py             # 规则引擎（规则判断 + 结果拆分）
├── run_all.sh                     # 一键运行脚本
└── README.md                      # 使用说明（本文档）

```

------

## 3. 核心设计思想

### 3.1 语义层与物理层解耦

- **rule.json**：只描述业务语义（如 `days_since_open`、`in_cnt_10d`）
- **CSV 特征文件**：存放具体实现字段（如 `days_since_open_from`）
- **规则引擎内部**：通过字段映射机制（`FIELD_ALIAS`）完成语义 → 物理字段解析

该设计避免规则与特征工程实现强耦合，便于后续扩展规则或调整特征。

------

## 4. 输入数据要求

### 4.1 原始交易数据（CSV）

原始交易数据需放置在 `data/` 目录中，至少包含以下字段：

- Timestamp
- From Account
- To Account
- Amount Paid
- Amount Received
- 其他业务相关字段

示例文件：

```
$AML_DATA_DIR/transactions.csv
```

------
### 4.2 抽样策略说明

- 抽样方式：随机抽样（uniform random sampling）

- 抽样位置：特征构建阶段

- 抽样目标：控制进入规则特征工程的数据规模

默认行为：

- 若数据行数 ≤ 抽样上限 → 使用全部数据

- 若数据行数 > 抽样上限 → 随机抽取指定数量样本

### 4.3 抽样参数

在 build_rule_features.py 中定义：

TARGET_N = 500_000


表示最多使用 50 万条交易记录 进行规则特征构建。

抽样过程使用固定随机种子，保证结果可复现。

## 5. 规则配置说明（rule.json）

规则通过 `rule.json` 配置，支持：

- 多规则定义（S1 / S3 / S5 / S6 / S7 等）
- 规则方向（IN / OUT）
- 前置条件（precondition）
- 规则条件（conditions）
- 比较算子：`>=`, `<=`, `==`, `between` 等

规则文件无需关心具体 CSV 字段名，只需描述业务语义。

------

## 6. 使用方法

### 6.1 一键运行（推荐）

#### 1️⃣ 使用默认数据集

```bash
./run_all.sh
```

默认使用：

```
data/detailed_transactions.csv
```

#### 2️⃣ 指定数据集（推荐用法）

```bash
./run_all.sh data/detailed_transactions.csv
```

也支持绝对路径：

```bash
./run_all.sh /path/to/your_dataset.csv
```

------

### 6.2 分步运行（可选）

#### 1️⃣ 构建规则特征

```bash
python build_rule_features.py
```

或指定输入文件：

```bash
python build_rule_features.py --input data/detailed_transactions.csv
```

输出：

```
out/txn_with_rule_features.csv
```

------

#### 2️⃣ 执行规则引擎

```bash
python run_rule_engine.py
```

输出：

```
out/txn_with_rule_hits.csv
```

------

## 7. 输出结果说明

### 7.1 规则特征文件

**文件**：`out/txn_with_rule_features.csv`

- 包含原始交易字段
- 包含各规则相关的中间特征（时间窗口统计、金额统计等）

------

### 7.2 规则命中结果

**文件**：`out/txn_with_rule_hits.csv`

新增字段示例：

```text
hit_S1
hit_S3
hit_S5
hit_S6
hit_S7
```

- `1`：该交易命中对应规则
- `0`：未命中

------

## 8. 结果校验（推荐）

可使用以下代码快速统计各规则命中数量：

```python
import pandas as pd

df = pd.read_csv("out/txn_with_rule_hits.csv")
hit_cols = [c for c in df.columns if c.startswith("hit_")]
print(df[hit_cols].sum().sort_values(ascending=False))
```

------

## 9. 自动化脚本说明（run_all.sh）

`run_all.sh` 负责：

- 接收命令行参数（数据集路径）
- 调度特征构建与规则引擎
- 出错即停止（`set -e`）
