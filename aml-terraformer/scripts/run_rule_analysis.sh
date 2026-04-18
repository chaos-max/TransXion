#!/bin/bash

# 规则命中分析脚本
# 在完整数据集上运行规则引擎并统计命中情况
#
# 使用前设置环境变量:
#   export AML_DATA_DIR=/path/to/your/data
#   export AML_ROOT=/path/to/aml-terraformer

echo "========================================"
echo "规则命中分析"
echo "========================================"
echo ""

# 设置默认值（如果未设置）
AML_ROOT=${AML_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
AML_DATA_DIR=${AML_DATA_DIR:-data}

cd "$AML_ROOT"

# 基本分析（只输出 JSON）
echo "运行规则命中分析..."
python scripts/analyze_rule_hits.py \
  --accounts "$AML_DATA_DIR/accounts.csv" \
  --transactions "$AML_DATA_DIR/transactions.csv" \
  --output output/rule_analysis/rule_hits_summary.json

echo ""
echo "========================================"
echo "分析完成！"
echo "========================================"
echo ""
echo "结果文件: output/rule_analysis/rule_hits_summary.json"
echo ""
echo "如需保存详细的 CSV 数据（包含每笔交易的规则命中标记），运行："
echo "  python scripts/analyze_rule_hits.py \\"
echo "    --accounts \$AML_DATA_DIR/accounts.csv \\"
echo "    --transactions \$AML_DATA_DIR/transactions.csv \\"
echo "    --output output/rule_analysis/rule_hits_summary.json \\"
echo "    --save-detailed-csv output/rule_analysis/transactions_with_hits.csv"
