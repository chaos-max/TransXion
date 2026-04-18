#!/usr/bin/env python3
"""测试异步初始化"""
import os
import sys
import asyncio
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import agentscope
from LLMGraph.environments.transaction import TransactionEnvironment
from LLMGraph.llms.load_configs import load_model_configs

async def test_async_init():
    """测试异步初始化"""
    print("=" * 70)
    print("测试异步初始化")
    print("=" * 70)

    # 配置路径
    config_path = os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
    output_dir = os.path.join(project_root, "transaction_output")

    # 加载配置
    print("\n1. 加载配置文件...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    time_cfg = env_cfg["time_configs"]
    txn_cfg = env_cfg["transaction_configs"]
    mgr_cfg = env_cfg["managers"]["transaction"]

    # 解析时间
    start_time = datetime.strptime(time_cfg["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(time_cfg["end_time"], "%Y-%m-%d %H:%M:%S")

    print(f"   开始时间: {start_time}")
    print(f"   结束时间: {end_time}")

    # 初始化 AgentScope
    print("\n2. 初始化 AgentScope...")
    model_configs = load_model_configs(
        model_config_path=os.path.join(project_root, "LLMGraph/llms/default_model_configs.json"),
        task_config_path=os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
    )
    agentscope.init(
        project="transaction_generation",
        name="test_async_init",
        model_configs=model_configs,
        use_monitor=False,
        save_code=False,
        save_api_invoke=False,
    )
    print("   ✓ AgentScope 初始化完成")

    # 创建环境
    print("\n3. 创建交易生成环境...")
    env_config = {
        "time_configs": {
            "start_time": start_time,
            "cur_time": start_time,
            "end_time": end_time,
            "window_size_minutes": time_cfg["window_size_minutes"],
        },
        "transaction_configs": txn_cfg,
        "managers": {
            "transaction": {
                **mgr_cfg,
                "generated_data_dir": output_dir,
            }
        },
        "task_path": project_root,
        "config_path": project_root,
    }

    env = TransactionEnvironment(**env_config)
    print("   ✓ 环境对象创建完成")

    # 异步初始化
    print("\n4. 异步初始化环境（这可能需要一些时间）...")
    await env.initialize_async()
    print("   ✓ 环境初始化完成")

    # 检查结果
    print("\n5. 检查初始化结果...")
    print(f"   账户总数: {len(env.manager_agent.manager.accounts)}")
    print(f"   个人账户: {len(env.manager_agent.person_ids)}")
    print(f"   商户账户: {len(env.manager_agent.merchant_ids)}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_async_init())
