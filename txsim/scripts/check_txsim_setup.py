#!/usr/bin/env python3
"""
Transaction场景测试脚本
用于验证配置、数据和模块是否正常
"""
import os
import sys
import json
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 项目根目录是 tests 的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """测试模块导入"""
    print("[1/4] 测试模块导入...")
    try:
        import agentscope
        from LLMGraph.environments.transaction import TransactionEnvironment
        from LLMGraph.manager.transaction import TransactionManager
        from LLMGraph.utils.data_generator import TransactionDataGenerator
        print("  ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置文件"""
    print("\n[2/4] 检查配置文件...")
    try:
        config_path = os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        txn_cfg = config["environment"]["transaction_configs"]
        print(f"  - num_merchants: {txn_cfg.get('num_merchants', 'Not set')}")
        print(f"  - num_persons: {txn_cfg.get('num_persons', 'Not set')}")
        print(f"  - target_txn_count: {txn_cfg.get('target_txn_count')}")
        print("  ✓ 配置文件正常")
        return True
    except Exception as e:
        print(f"  ✗ 配置文件错误: {e}")
        return False


def test_data_files():
    """检查数据文件"""
    print("\n[3/4] 检查数据文件...")
    data_dir = os.path.join(project_root, "LLMGraph/tasks/transaction/data")

    # 检查profiles
    profiles_path = os.path.join(data_dir, "profiles.json")
    if os.path.exists(profiles_path):
        with open(profiles_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                count = len(data["records"])
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 0
        print(f"  ✓ profiles.json: {count} 条记录")
    else:
        print(f"  ! profiles.json 不存在 (将在运行时生成)")

    # 检查merchants
    merchants_path = os.path.join(data_dir, "merchants.json")
    if os.path.exists(merchants_path):
        with open(merchants_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                count = len(data["records"])
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 0
        print(f"  ✓ merchants.json: {count} 条记录")
    else:
        print(f"  ! merchants.json 不存在 (将在运行时生成)")

    return True


def test_agentscope():
    """测试AgentScope初始化"""
    print("\n[4/4] 测试AgentScope...")
    try:
        import agentscope
        from LLMGraph.llms.load_configs import load_model_configs

        # 加载模型配置（与真实运行一致）
        model_configs = load_model_configs(
            model_config_path=os.path.join(project_root, "LLMGraph/llms/default_model_configs.json"),
            task_config_path=os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
        )

        agentscope.init(
            project="transaction_test",
            name="test",
            model_configs=model_configs,
            use_monitor=False,
            save_code=False,
            save_api_invoke=False,
        )
        print("  ✓ AgentScope 初始化成功")
        return True
    except Exception as e:
        print(f"  ✗ AgentScope 初始化失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Transaction 场景测试")
    print("=" * 60)

    results = [
        test_imports(),
        test_config(),
        test_data_files(),
        test_agentscope()
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ 所有测试通过,系统已就绪!")
        print("\n运行主程序: python main_txn_async.py")
    else:
        print("✗ 部分测试失败,请检查配置")
        sys.exit(1)
    print("=" * 60)
