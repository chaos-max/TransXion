"""
TransactionManagerAgent: TransactionManager 的 Agent 封装

职责：
- 持有并转发调用 TransactionManager（权威状态机）
- 提供 Msg 路由接口 reply()
- 提供便捷方法供 TransactionAgent/Environment 调用
- 不重复实现任何业务逻辑，只做转发与包装
"""

from typing import Optional, Dict, List, Any
from datetime import datetime

from agentscope.agents import AgentBase
from agentscope.message import Msg

from LLMGraph.agent import manager_agent_registry
from LLMGraph.manager.transaction import TransactionManager


@manager_agent_registry.register("transaction_manager")
class TransactionManagerAgent(AgentBase):
    """TransactionManager 的 Agent 包装器"""

    def __init__(
        self,
        name: str,
        model_config_name: Optional[str] = None,
        transaction_data_dir: Optional[str] = None,
        generated_data_dir: Optional[str] = None,
        transaction_manager_configs: Optional[Dict[str, Any]] = None,
        cur_time: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 TransactionManagerAgent

        Args:
            name: agent 名称
            model_config_name: 模型配置名（可选，AgentBase 需要）
            transaction_data_dir: 交易数据目录（profiles.json/merchants.json所在目录）
            generated_data_dir: 生成数据输出目录
            transaction_manager_configs: TransactionManager 配置（可选）
            cur_time: 当前时间字符串（可选）
            **kwargs: 其他 AgentBase 参数
        """
        super().__init__(name=name, model_config_name=model_config_name, **kwargs)

        self.transaction_data_dir = transaction_data_dir
        self.generated_data_dir = generated_data_dir
        self.cur_time = cur_time
        self.transaction_manager_configs = transaction_manager_configs or {}

        # TransactionManager 实例（延迟初始化）
        self.manager: Optional[TransactionManager] = None

    def reply(self, message: Msg) -> Msg:
        """
        处理消息并返回响应

        支持的命令：
        - "load_data": 初始化 TransactionManager
        - "is_rerun": 检查是否续跑
        - "call_manager_func": 调用 manager 方法
        - 其他：返回错误信息

        Args:
            message: 输入消息

        Returns:
            响应消息
        """
        content = message.content

        try:
            if content == "load_data":
                return Msg(
                    name=self.name,
                    role="assistant",
                    content="Error: load_data is not supported in sync mode. Use load_data_async instead."
                )
            elif content == "is_rerun":
                return self._handle_is_rerun(message)
            elif content == "call_manager_func":
                return self._handle_call_manager_func(message)
            else:
                return Msg(
                    name=self.name,
                    role="assistant",
                    content=f"Unknown command: {content}"
                )
        except Exception as e:
            return Msg(
                name=self.name,
                role="assistant",
                content=f"Error processing command '{content}': {str(e)}"
            )


    async def _handle_load_data_async(self, message: Msg) -> Msg:
        """处理 load_data 命令（异步版本）"""
        kwargs = message.get("kwargs", {})
        start_time = kwargs.pop("start_time", None)
        end_time = kwargs.pop("end_time", None)

        # 合并配置
        import os
        manager_kwargs = {**self.transaction_manager_configs}

        # 设置数据路径
        if self.generated_data_dir:
            manager_kwargs["output_dir"] = self.generated_data_dir

        if self.transaction_data_dir:
            profiles_path = os.path.join(self.transaction_data_dir, "profiles.json")
            merchants_path = os.path.join(self.transaction_data_dir, "merchants.json")
            if os.path.exists(profiles_path):
                manager_kwargs["profiles_path"] = profiles_path
            if os.path.exists(merchants_path):
                manager_kwargs["merchants_path"] = merchants_path

        # 合并消息中的额外参数（start_time 和 end_time 已经被 pop 掉了）
        manager_kwargs.update(kwargs)

        # 调用 TransactionManager.load_data_async（异步版本）
        self.manager = await TransactionManager.load_data_async(
            start_time=start_time,
            end_time=end_time,
            **manager_kwargs
        )

        return Msg(
            name=self.name,
            role="assistant",
            content="Data loaded successfully"
        )

    def _handle_is_rerun(self, message: Msg) -> Msg:
        """处理 is_rerun 命令"""
        if self.manager is not None:
            is_rerun = self.manager.rerun()
        else:
            # 临时构造 manager 检查
            output_dir = message.get("output_dir") or self.transaction_manager_configs.get("output_dir")
            if output_dir:
                import os
                state_file = os.path.join(output_dir, "state.json")
                is_rerun = os.path.exists(state_file) and os.path.getsize(state_file) > 0
            else:
                is_rerun = False

        return Msg(
            name=self.name,
            role="assistant",
            content=is_rerun
        )

    def _handle_call_manager_func(self, message: Msg) -> Msg:
        """处理 call_manager_func 命令"""
        if self.manager is None:
            return Msg(
                name=self.name,
                role="assistant",
                content="Manager not initialized. Call 'load_data' first."
            )

        func_name = message.get("func")
        kwargs = message.get("kwargs", {})

        if not func_name:
            return Msg(
                name=self.name,
                role="assistant",
                content="Missing 'func' parameter"
            )

        if not hasattr(self.manager, func_name):
            return Msg(
                name=self.name,
                role="assistant",
                content=f"Manager has no method '{func_name}'"
            )

        try:
            method = getattr(self.manager, func_name)
            result = method(**kwargs)

            return Msg(
                name=self.name,
                role="assistant",
                content=result
            )
        except Exception as e:
            return Msg(
                name=self.name,
                role="assistant",
                content=f"Error calling {func_name}: {str(e)}"
            )

    # ==================== 便捷封装方法 ====================

    def get_geo_person_ids(self, geo_id: int) -> List[str]:
        """
        获取指定 geo 内的个人账户 ID 列表

        Args:
            geo_id: geo ID

        Returns:
            个人账户 ID 列表
        """
        if self.manager is None:
            return []
        return self.manager.geo_persons.get(geo_id, [])

    def get_geo_merchant_ids(self, geo_id: int) -> List[str]:
        """
        获取指定 geo 内的商户账户 ID 列表

        Args:
            geo_id: geo ID

        Returns:
            商户账户 ID 列表
        """
        if self.manager is None:
            return []
        return self.manager.geo_merchants.get(geo_id, [])

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """
        获取账户信息

        Args:
            account_id: 账户 ID

        Returns:
            账户信息字典，或 None
        """
        if self.manager is None:
            return None
        return self.manager.account_map.get(account_id)

    def choose_counterparty(
        self,
        from_id: str,
        to_merchant: Optional[bool] = None,
        cross: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        为指定账户选择交易对手

        Args:
            from_id: 发起方账户 ID
            to_merchant: 对手是否为商户（保留参数，实际由manager根据from_type自动决定）
            cross: 是否跨 geo（None=自动决定）

        Returns:
            对手账户字典，或 None
        """
        if self.manager is None:
            return None

        from_account = self.manager.account_map.get(from_id)
        if from_account is None:
            return None

        # 直接调用 manager.choose_counterparty
        # manager 会根据 from_type 自动决定 to_merchant
        return self.manager.choose_counterparty(from_account, cross=cross)

    def sample_senders(self, txn_count: int) -> List[str]:
        """
        采样发起方账户

        Args:
            txn_count: 交易数量

        Returns:
            发起方账户 ID 列表
        """
        if self.manager is None:
            return []
        return self.manager.sample_senders(txn_count)

    def allocate_budget(
        self,
        window_start: datetime,
        window_end: datetime,
        total_events: int
    ) -> Dict[int, int]:
        """
        按 geo 分配交易预算

        Args:
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            total_events: 总交易数

        Returns:
            geo_id -> 分配的交易数
        """
        if self.manager is None:
            return {}
        return self.manager.allocate_budget(window_start, window_end, total_events)

    def apply_events(self, events: List[Dict[str, Any]]) -> int:
        """
        应用（落地）交易事件

        Args:
            events: 交易事件列表

        Returns:
            成功添加的交易数
        """
        if self.manager is None:
            return 0
        return self.manager.apply_events(events)

    def save_infos(
        self,
        cur_time: datetime,
        start_time: datetime,
        force: bool = False
    ) -> None:
        """
        保存状态和日志

        Args:
            cur_time: 当前时间
            start_time: 开始时间
            force: 是否强制保存
        """
        if self.manager is None:
            return
        self.manager.save_infos(cur_time, start_time, force=force)

    def get_start_time(self) -> str:
        """
        获取开始时间字符串

        Returns:
            开始时间字符串
        """
        if self.manager is None:
            return ""
        return self.manager.get_start_time()

    def sample_timestamp(
        self,
        window_start: datetime,
        window_end: datetime,
        tod_p: List[float]
    ) -> datetime:
        """
        采样时间戳

        Args:
            window_start: 窗口开始
            window_end: 窗口结束
            tod_p: 时间分布

        Returns:
            采样的时间戳
        """
        if self.manager is None:
            return window_start
        return self.manager.sample_timestamp(window_start, window_end, tod_p)

    def sample_amounts(
        self,
        from_account: Dict[str, Any],
        to_account: Dict[str, Any]
    ) -> tuple:
        """
        采样交易金额

        Args:
            from_account: 发起方账户
            to_account: 对手账户

        Returns:
            (amount_paid, amount_received)
        """
        if self.manager is None:
            return (0.0, 0.0)
        return self.manager.sample_amounts(from_account, to_account)

    def sample_payment_format(self) -> str:
        """
        采样支付方式

        Returns:
            支付方式字符串
        """
        if self.manager is None:
            return "Mobile"  # 默认使用移动支付
        return self.manager.sample_payment_format()

    def add_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """
        批量添加交易

        Args:
            transactions: 交易列表

        Returns:
            成功添加的交易数
        """
        if self.manager is None:
            return 0
        return self.manager.add_transactions(transactions)

    def update_top_merchants(self, top_k: int = 20) -> None:
        """
        更新 top merchants 列表

        Args:
            top_k: top K 数量
        """
        if self.manager is None:
            return
        self.manager.update_top_merchants(top_k)

    @property
    def num_geos(self) -> int:
        """获取 geo 数量"""
        if self.manager is None:
            return 0
        return self.manager.num_geos

    @property
    def person_ids(self) -> List[str]:
        """获取所有个人账户 ID"""
        if self.manager is None:
            return []
        return self.manager.person_ids

    @property
    def merchant_ids(self) -> List[str]:
        """获取所有商户账户 ID"""
        if self.manager is None:
            return []
        return self.manager.merchant_ids

    @property
    def total_txn_generated(self) -> int:
        """获取已生成的交易总数"""
        if self.manager is None:
            return 0
        return self.manager.total_txn_generated
