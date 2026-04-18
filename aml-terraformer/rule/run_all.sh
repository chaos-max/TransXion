#!/bin/bash
set -e

# ===============================
# 用法说明
# ===============================
# export AML_DATA_DIR=/path/to/your/data   # 设置数据目录
# ./run_all.sh                             # 使用 $AML_DATA_DIR/transactions.csv
# ./run_all.sh /path/to/custom_data.csv    # 使用指定文件
# ===============================

INPUT_DATASET=$1

# 如果没有指定数据集，使用环境变量
if [ -z "$INPUT_DATASET" ]; then
    if [ -z "$AML_DATA_DIR" ]; then
        echo "[ERROR] Please set AML_DATA_DIR or specify input dataset"
        exit 1
    fi
    INPUT_DATASET="$AML_DATA_DIR/transactions.csv"
    echo "[INFO] Using AML_DATA_DIR: $INPUT_DATASET"
fi

echo "=== Step 1: Build rule features ==="
python build_rule_features.py --input "$INPUT_DATASET"

echo "=== Step 2: Run rule engine ==="
python run_rule_engine.py

echo "=== All done ==="
