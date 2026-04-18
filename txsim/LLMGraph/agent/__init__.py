from LLMGraph.registry import Registry
agent_registry = Registry(name="AgentRegistry")
manager_agent_registry = Registry(name="ManagerAgentRegistry")


# from .tool_agent import ToolAgent  # 注释掉：transaction 场景不需要
# from .general import GeneralAgent  # 注释掉：transaction 场景不需要
from .transaction import TransactionAgent,TransactionManagerAgent