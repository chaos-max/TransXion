"""
增强版配置加载器：支持从任务配置文件读取 role_mappings

使用方法：
    from LLMGraph.llms.load_configs_v2 import load_model_configs_with_task
    configs = load_model_configs_with_task(task_config_path='path/to/config.yaml')
"""
import json
import os
import yaml
from typing import Optional, Dict, List


def load_model_configs_with_task(
    model_config_path: Optional[str] = None,
    task_config_path: Optional[str] = None,
    role_mappings_override: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> List[dict]:
    """
    加载模型配置，支持从任务配置文件读取 role_mappings

    Args:
        model_config_path: 模型配置文件路径（JSON），默认为 default_model_configs.json
        task_config_path: 任务配置文件路径（YAML），从中读取 llm_role_mappings
        role_mappings_override: 直接提供的 role_mappings，优先级最高
        verbose: 是否打印配置信息

    Returns:
        list: AgentScope 兼容的配置列表
    """
    # 1. 加载模型基础配置
    if model_config_path is None:
        model_config_path = os.path.join(
            os.path.dirname(__file__),
            "default_model_configs.json"
        )

    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    # 如果是旧格式（直接是列表），直接返回
    if isinstance(model_data, list):
        return model_data

    # 2. 确定 role_mappings 来源（优先级：override > task_config > model_config）
    role_mappings = {}

    # 从模型配置文件读取（最低优先级）
    if 'role_mappings' in model_data:
        role_mappings = model_data['role_mappings'].copy()
        if verbose:
            print(f"📋 从模型配置文件加载 role_mappings")

    # 从任务配置文件读取（中等优先级）
    if task_config_path and os.path.exists(task_config_path):
        with open(task_config_path, 'r', encoding='utf-8') as f:
            task_data = yaml.safe_load(f)

        if task_data and 'llm_role_mappings' in task_data:
            role_mappings.update(task_data['llm_role_mappings'])
            if verbose:
                print(f"📋 从任务配置文件加载 llm_role_mappings: {task_config_path}")

    # 直接提供的覆盖（最高优先级）
    if role_mappings_override:
        role_mappings.update(role_mappings_override)
        if verbose:
            print(f"📋 使用提供的 role_mappings 覆盖")

    # 3. 生成角色配置
    base_configs = {c['config_name']: c for c in model_data['model_configs']}
    generated_configs = []

    if verbose and role_mappings:
        print("\n" + "="*70)
        print("🤖 LLM 角色映射配置")
        print("="*70)

    for role, base_config_name in role_mappings.items():
        if base_config_name in base_configs:
            role_config = base_configs[base_config_name].copy()
            role_config['config_name'] = role

            # 深拷贝嵌套字典
            if 'client_args' in role_config:
                role_config['client_args'] = role_config['client_args'].copy()
            if 'generate_args' in role_config:
                role_config['generate_args'] = role_config['generate_args'].copy()

            generated_configs.append(role_config)

            # 打印角色映射信息
            if verbose:
                model_name = role_config.get('model_name', 'Unknown')
                model_type = role_config.get('model_type', 'Unknown')
                base_url = role_config.get('client_args', {}).get('base_url', 'N/A')

                print(f"  • {role:25s} → {base_config_name:20s}")
                print(f"    ├─ Model: {model_name}")
                print(f"    ├─ Type:  {model_type}")
                print(f"    └─ URL:   {base_url}")
        else:
            if verbose:
                print(f"  ⚠️  警告: 角色 '{role}' 映射到不存在的配置 '{base_config_name}'")

    if verbose and role_mappings:
        print("="*70 + "\n")

    # 返回：角色配置 + 底层配置
    return generated_configs + model_data['model_configs']


def print_role_mapping_info(role: str, config_name: str):
    """
    在运行时打印角色使用的LLM配置信息

    Args:
        role: 角色名称
        config_name: 配置名称
    """
    print(f"🔧 [{role}] 使用配置: {config_name}")


# Alias for backward compatibility
load_model_configs = load_model_configs_with_task
