# 规则命中分析工具

## 功能

这个工具在完整数据集上运行 rule-based monitor，统计每个规则的命中情况。

## 输出内容

### 1. 每个规则的命中数量（所有交易）
- 总命中数
- 命中率

### 2. 异常交易中每个规则的命中率
- 异常交易总数
- 每个规则在异常交易中的命中数
- 每个规则在异常交易中的命中率（召回率）

### 3. 规则性能指标
- **精确率 (Precision)**: 命中的交易中有多少是真的异常
- **召回率 (Recall)**: 异常交易中有多少被这个规则命中

### 4. 规则组合分析
- 每笔交易命中了多少个规则的分布
- 异常交易和正常交易的对比

## 使用方法

### 快速运行

```bash
bash scripts/run_rule_analysis.sh
```

### 完整命令

```bash
python scripts/analyze_rule_hits.py \
  --accounts data/account-reference.csv \
  --transactions data/trans-reference.csv \
  --output output/rule_analysis/rule_hits_summary.json
```

### 保存详细数据（可选）

如果你想保存每笔交易的规则命中标记到 CSV：

```bash
python scripts/analyze_rule_hits.py \
  --accounts data/account-reference.csv \
  --transactions data/trans-reference.csv \
  --output output/rule_analysis/rule_hits_summary.json \
  --save-detailed-csv output/rule_analysis/transactions_with_hits.csv
```

详细 CSV 包含：
- 所有原始交易列
- `hit_S1`, `hit_S3`, `hit_S5`, `hit_S6`, `hit_S7`, `hit_S8`: 每个规则的命中标记（0或1）
- `total_rule_hits`: 该交易命中的规则总数

## 输出格式

### JSON 结构

```json
{
  "summary": {
    "total_transactions": 5078345,
    "laundering_transactions": 50783,
    "normal_transactions": 5027562,
    "laundering_ratio": 0.01
  },
  "rules": {
    "S1": {
      "rule_name": "新账户立即大额转出",
      "direction": "OUT",
      "all_transactions": {
        "total_hits": 12345,
        "hit_rate": 0.0024
      },
      "laundering_transactions": {
        "total_hits": 8234,
        "hit_rate": 0.1621,
        "total_laundering": 50783
      },
      "normal_transactions": {
        "total_hits": 4111,
        "hit_rate": 0.0008,
        "total_normal": 5027562
      },
      "metrics": {
        "precision": 0.6669,
        "recall": 0.1621
      }
    },
    "S3": { ... },
    ...
  },
  "rule_combination": {
    "laundering": {
      "0": 5234,
      "1": 12456,
      "2": 18234,
      "3": 10123,
      ...
    },
    "normal": {
      "0": 4523456,
      "1": 423456,
      ...
    }
  },
  "average_hits": {
    "laundering": 2.34,
    "normal": 0.12
  }
}
```

## 输出解读

### 命令行输出示例

```
【S1】新账户立即大额转出
  方向: OUT
  所有交易:
    命中数: 12,345 / 5,078,345 (0.24%)
  异常交易:
    命中数: 8,234 / 50,783 (16.21%)
  正常交易:
    命中数: 4,111 / 5,027,562 (0.08%)
  性能指标:
    精确率 (Precision): 66.69% (命中的交易中有多少是真异常)
    召回率 (Recall):    16.21% (异常交易中有多少被命中)
```

### 关键指标解释

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **总命中数** | 规则触发的总次数 | - |
| **异常交易命中率 (召回率)** | 异常交易中被规则捕获的比例 | 越高越好 |
| **精确率** | 规则命中的交易中真的异常的比例 | 越高越好 |
| **正常交易命中率** | 正常交易被误判的比例 | 越低越好 (减少误报) |

### 规则性能评价标准

```
优秀规则:
  - 召回率高 (>30%)
  - 精确率高 (>70%)
  - 正常交易命中率低 (<1%)

一般规则:
  - 召回率中等 (10-30%)
  - 精确率中等 (40-70%)

待改进规则:
  - 召回率低 (<10%)
  - 精确率低 (<40%)
  - 正常交易命中率高 (>5%)
```

## 规则列表

当前系统包含的规则（来自 rule/data/rule.json）：

- **S1**: 新账户立即大额转出
- **S3**: 短期流入后快速流出
- **S5**: 交易量递增模式
- **S6**: 快进快出（资金过境）
- **S7**: 交易规律性模式
- **S8**: 多源资金聚集

## 性能说明

- **数据量**: 约 500 万笔交易
- **运行时间**: 约 3-5 分钟（取决于机器性能）
- **内存需求**: 约 4-6 GB

## 注意事项

1. **数据必须已标注**: 需要 `Is Laundering` 列来区分异常/正常交易
2. **规则顺序**: 不同规则可能有依赖关系，输出顺序与配置文件一致
3. **规则组合**: 一笔交易可以同时命中多个规则

## 后续分析建议

1. **规则优化**: 找出召回率低的规则进行改进
2. **阈值调整**: 基于精确率和召回率调整规则参数
3. **规则组合**: 研究哪些规则组合最有效
4. **对比实验**: 用这个作为 baseline，对比模型训练前后的变化

## 示例：查看最有效的规则

运行后查看 JSON 输出，找 `metrics.recall` 最高的规则：

```bash
# 查看规则性能排名
cat output/rule_analysis/rule_hits_summary.json | \
  jq -r '.rules | to_entries | sort_by(.value.metrics.recall) | reverse | .[] | "\(.key): \(.value.metrics.recall * 100)%"'
```

## 问题排查

### 错误: KeyError: 'Is Laundering'
数据中没有异常标注列，需要确保数据已经过预处理。

### 运行时间过长
数据量很大时正常，可以：
1. 先用小样本测试
2. 在后台运行: `nohup bash scripts/run_rule_analysis.sh &`

### 内存不足
减少数据量或增加系统内存。
