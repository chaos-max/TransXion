from pydantic import BaseModel

from typing import Any, Optional, Union, Sequence, Literal
from abc import abstractmethod

from agentscope.message import MessageBase
from agentscope.models import ModelResponse
import json


class AgentOutputParser(BaseModel):
    
    # @abstractmethod
    def parse(self, text: str) -> Union[dict,list,str]:
        """Parse text into json_field."""

    def parse_func(self, output:ModelResponse) -> ModelResponse:
        raw_data = output.text
        try:
            json_field = self.parse(raw_data)
            
        except Exception as e:
            json_field = {"fail":True}

        output.json = json_field
        return output
    
def find_and_load_json(s, outer_type = "dict"):
    """
    从字符串中提取并解析 JSON
    支持:
    - 纯 JSON 字符串
    - Markdown 代码块包裹的 JSON
    - 带有额外文本的 JSON
    - 带有注释的 JSON (去除 // 和 /* */ 注释)
    """
    import re

    # 1. 先尝试直接解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2. 提取 Markdown 代码块中的内容
    # 匹配 ```json ... ``` 或 ``` ... ```
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_blocks = re.findall(code_block_pattern, s, re.DOTALL)

    if code_blocks:
        # 使用第一个代码块
        s = code_blocks[0]

    # 3. 移除 JSON 中的注释
    # 移除单行注释 // ...
    s = re.sub(r'//.*?(?=\n|$)', '', s)
    # 移除多行注释 /* ... */
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    # 4. 根据类型查找 JSON
    if outer_type == "dict":
        start_pos = s.find('{')
        end_pos = s.rfind('}')
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            json_str = s[start_pos:end_pos+1]
            # 尝试解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试修复常见问题
                try:
                    # 可能缺少结尾
                    return json.loads(json_str + "}")
                except json.JSONDecodeError:
                    # 返回原始字符串
                    return s

    elif outer_type == "list":
        start_pos = s.find('[')
        end_pos = s.rfind(']')
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            json_str = s[start_pos:end_pos+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                import ast
                try:
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    try:
                        return ast.literal_eval(json_str + "]")
                    except (ValueError, SyntaxError):
                        try:
                            return ast.literal_eval("[" + json_str + "]")
                        except (ValueError, SyntaxError):
                            return s

    return s