from typing import Dict
from LLMGraph.registry import Registry
env_registry = Registry(name="EnvironmentRegistry")
from .base import BaseEnvironment
# from .general import GeneralEnvironment  # 注释掉：transaction 场景不需要
from .transaction import TransactionEnvironment