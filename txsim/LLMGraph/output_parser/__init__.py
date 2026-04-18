"""
输出解析器模块

这个模块包含了不同类型的输出解析器注册表:

- output_parser_registry: 通用输出解析器注册表
- control_output_parser_registry: 控制相关输出解析器注册表
- general_output_parser_registry: 通用输出解析器注册表
"""

from LLMGraph.registry import Registry

output_parser_registry = Registry(name="OutputParserRegistry")
control_output_parser_registry = Registry(name="ControlOutputParserRegistry")
general_output_parser_registry = Registry(name="GeneralOutputParserRegistry")

from .control import *
from .general import *