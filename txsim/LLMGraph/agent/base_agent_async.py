"""Async version of BaseGraphAgent with parallel execution support."""
import asyncio
import json
from typing import List, Optional, Dict, Any
from loguru import logger

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.memory import TemporaryMemory
from LLMGraph.output_parser.base_parser import AgentOutputParser
from LLMGraph.utils.count import parse_prompt


class AsyncBaseGraphAgent(AgentBase):
    """
    异步版本的 BaseGraphAgent，支持并行模型调用。

    主要优化：
    1. 支持异步模型调用（如果模型支持）
    2. 可以批量并行处理多个请求
    3. 减少等待时间
    """

    def __init__(
        self,
        name: str,
        prompt_template,
        output_parser: AgentOutputParser,
        max_retrys: int = 2,
        sys_prompt: str = None,
        model_config_name: str = None,
        use_memory: bool = True,
        memory_config: dict = None,
        to_dist=False,
        short_memory_config: Optional[dict] = None,
    ) -> None:

        super().__init__(
            name, sys_prompt, model_config_name, use_memory, memory_config, to_dist
        )

        self.prompt_template = prompt_template
        self.max_retrys = max_retrys
        self.short_memory = TemporaryMemory(short_memory_config)
        self.output_parser = output_parser

    async def step_async(self, prompt_inputs) -> Msg:
        """异步版本的 step 方法"""

        desearlize_prompt_msgs = [
            *self.prompt_template.format_messages(**prompt_inputs),
            *self.short_memory.get_memory(recent_n=3),
        ]

        for msg in desearlize_prompt_msgs:
            if isinstance(msg.name, int):
                msg.name = str(msg.name)

        prompt = self.model.format(desearlize_prompt_msgs)

        # 尝试多次重试
        for i in range(self.max_retrys):
            prompt = parse_prompt(prompt)

            # 在线程池中执行模型调用（如果模型不支持异步）
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: self.model(
                    prompt,
                    parse_func=self.output_parser.parse_func,
                    max_retries=self.max_retrys,
                ),
            )

            res_json = res.json
            if res_json and not res_json.get("fail", False):
                msg_finish = Msg(self.name, res.json, role="assistant")
                self.speak(
                    Msg(self.name, json.dumps(res.raw, indent=4), role="assistant"),
                )
                return msg_finish

        return Msg("user", content=f"FAIL TO PARSE", role="user", fail=True)

    async def reply_async(self, message: Msg = None) -> Msg:
        """异步版本的 reply 方法"""

        if message.content == "get_prompt":
            kwargs = message.get("kwargs", {})
            func_name = message.get("func", "")
            func = getattr(self, func_name)

            # 在线程池中执行函数
            loop = asyncio.get_event_loop()
            prompt_inputs = await loop.run_in_executor(None, lambda: func(**kwargs))

            return_msgs = self.prompt_template.format_messages(**prompt_inputs)
            return Msg("user", content=return_msgs, role="user")

        if message.content == "call_function":
            func_name = message.get("func", "")
            kwargs = message.get("kwargs", {})
            func = getattr(self, func_name)

            # 在线程池中执行函数
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, lambda: func(**kwargs))

            if res is None:
                res = Msg(self.name, content=None, role="assistant")
            assert isinstance(res, Msg), res
            return res

        if message.content == "reply_prompt":
            prompt_msgs = message.get("prompt", [])
            desearlize_prompt_msgs = []

            for prompt_msg in prompt_msgs:
                if isinstance(prompt_msg, Msg):
                    desearlize_prompt_msgs.append(prompt_msg)
                elif isinstance(prompt_msg, dict):
                    desearlize_prompt_msgs.append(Msg(**prompt_msg))

            # parse name int
            for msg in desearlize_prompt_msgs:
                if isinstance(msg.name, int):
                    msg.name = str(msg.name)

            prompt = self.model.format(*desearlize_prompt_msgs)
            prompt = parse_prompt(prompt)

            try:
                # 在线程池中执行模型调用
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None,
                    lambda: self.model(
                        prompt,
                        parse_func=self.output_parser.parse_func,
                        max_retries=3,
                    ),
                )

                output = res.json
                if "fail" in output.keys():
                    return Msg(
                        "user", content=f"FAIL TO PARSE", role="user", fail=True
                    )

                if "actions" in output.keys():
                    actions = output["actions"]

                    if len(actions) > 5:
                        actions = actions[:5]  # 每次function call 最多同时执行五个

                    call_tool_msg = Msg(
                        self.name,
                        role="assistant",
                        func="call_tool_func",
                        content=actions,
                        finish=False,
                    )
                    return call_tool_msg

                elif "return_values" in output.keys():
                    msg_finish = Msg(self.name, output, role="assistant", finish=True)
                    self.speak(
                        Msg(
                            self.name, json.dumps(res.raw, indent=4), role="assistant"
                        ),
                    )
                    return msg_finish
                else:
                    return Msg(
                        "user", content=f"FAIL TO PARSE", role="user", fail=True
                    )
            except Exception as e:
                logger.error(f"Reply async failed: {e}")
                return Msg("user", content=f"FAIL TO PARSE", role="user", fail=True)

        return Msg("user", content=f"FAIL TO PARSE", role="user", fail=True)

    def step(self, prompt_inputs) -> Msg:
        """同步入口，调用异步方法"""
        return asyncio.run(self.step_async(prompt_inputs))

    def reply(self, message: Msg = None) -> Msg:
        """同步入口，调用异步方法"""
        return asyncio.run(self.reply_async(message))

    def get_short_memory(self) -> Msg:
        memory = self.short_memory.get_memory(recent_n=3)
        return Msg(self.name, content=memory, role="assistant")

    def clear_short_memory(self):
        self.short_memory.clear()
