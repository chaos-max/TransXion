"""
TransactionEnvironment: 资金交易社会模拟环境

用于生成百万级交易事件的规则驱动环境（不使用LLM agent）。
系统规模：约10,000账户（个人+商户混合），跨度一年，时间戳精度到秒。

架构说明：
- TransactionManager: 负责数据管理、状态维护、统计、保存（在 LLMGraph/manager/transaction.py）
- TransactionManagerAgent: 负责 Msg 路由和薄封装（在 LLMGraph/agent/transaction/manager_agent.py）
- TransactionAgent: 负责子图级交易生成（在 LLMGraph/agent/transaction/transaction_agent.py）
- TransactionEnvironment: 负责调度、step 推进、调用 manager/agent 接口
"""

import random
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any

from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
from LLMGraph.agent.transaction.manager_agent import TransactionManagerAgent
from LLMGraph.agent.transaction.transaction_agent import TransactionAgent
from LLMGraph.prompt.transaction import build_daily_scenario_prompt
from agentscope.message import Msg
from agentscope.models import ModelWrapperBase, load_model_by_config_name


@EnvironmentRegistry.register("transaction")
class TransactionEnvironment(BaseEnvironment):
    """交易环境类（调度器）"""

    time_configs: dict = {}
    transaction_configs: dict = {}
    output_dir: str = "./transaction_output"
    random_seed: int = 42
    transaction_agent: Any = None  # 子图级交易生成 agent
    daily_scenario_cache: Dict[date, dict] = {}  # 每日场景缓存
    model_config_name: str = "scenario_generator"  # LLM 模型配置名称（场景生成器）
    llm_logger: Any = None  # LLM 日志记录器

    # Pydantic 1.x 语法 - 继承自 BaseEnvironment
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """初始化交易环境"""
        # 提取配置（仿照 SocialEnvironment）
        managers_config = kwargs.pop("managers", {})
        transaction_manager_configs = managers_config.get("transaction", {})
        task_path = kwargs.pop("task_path", ".")
        config_path = kwargs.pop("config_path", ".")

        # 时间配置
        time_configs = kwargs.pop("time_configs", {})
        default_time_configs = {
            "start_time": datetime(2025, 1, 1, 0, 0, 0),
            "cur_time": datetime(2025, 1, 1, 0, 0, 0),
            "end_time": datetime(2026, 1, 1, 0, 0, 0),
            "window_size_minutes": 60,
        }
        default_time_configs.update(time_configs)

        # 交易配置
        transaction_configs_input = kwargs.pop("transaction_configs", {})
        default_transaction_configs = {
            "target_txn_count": 1_000_000,
            "num_accounts": 10_000,
            "merchant_ratio": 0.2,
            "num_merchants": None,  # 新增:优先使用这个
            "num_persons": None,    # 新增:优先使用这个
            "use_fixed_daily_generation": False,  # 新增:是否使用固定每日生成模式
            "p2m_ratio_for_person": 0.75,
            "m2m_ratio_for_merchant": 0.7,
            "exploration_alpha": 5.0,
            "favorites_topk": 10,
            "buffer_size": 10000,
        }
        default_transaction_configs.update(transaction_configs_input)

        # 提取银行配置（需要在创建 TransactionManagerAgent 之前）
        banks = default_transaction_configs.get("banks", ["ICBC", "CCB", "ABC", "BOC", "CMB"])

        # 输出目录
        transaction_data_dir = transaction_manager_configs.get("transaction_data_dir", "./transaction_data")
        generated_data_dir = transaction_manager_configs.get("generated_data_dir", "./transaction_output")

        if not os.path.isabs(generated_data_dir):
            generated_data_dir = os.path.join(os.path.dirname(config_path), generated_data_dir)

        output_dir = generated_data_dir
        random_seed = transaction_manager_configs.get("random_seed", 42)
        random.seed(random_seed)

        # 创建 TransactionManagerAgent
        manager_agent = TransactionManagerAgent(
            name="transaction_manager",
            model_config_name="vllm_local",  # 直接使用本地 vLLM 模型
            transaction_data_dir=transaction_data_dir,
            generated_data_dir=generated_data_dir,
            transaction_manager_configs={
                **default_transaction_configs,
                "random_seed": random_seed,
                "banks": banks,  # 传递银行配置
            },
            cur_time=default_time_configs["cur_time"].strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 调用父类初始化，传入所有字段值
        super().__init__(
            manager_agent=manager_agent,
            time_configs=default_time_configs,
            transaction_configs=default_transaction_configs,
            output_dir=output_dir,
            random_seed=random_seed,
            **kwargs
        )

        # 存储 manager_agent 引用
        self.manager_agent = manager_agent

        # 初始化每日场景缓存
        self.daily_scenario_cache = {}

        # 初始化 LLM 日志记录器
        self._setup_llm_logger(generated_data_dir)

        # 提取货币配置
        currencies = default_transaction_configs.get("currencies", ["USD"])
        default_currency = default_transaction_configs.get("default_currency", "USD")
        cross_currency_ratio = default_transaction_configs.get("cross_currency_ratio", 0.05)
        exchange_rates = default_transaction_configs.get("exchange_rates", {default_currency: 1.0})

        # 提取手续费配置
        fee_configs = default_transaction_configs.get("fee_configs", {})
        same_currency_fee_rate = fee_configs.get("same_currency_fee_rate", 0.001)
        cross_currency_fee_rate = fee_configs.get("cross_currency_fee_rate", 0.015)
        exchange_rate_spread = fee_configs.get("exchange_rate_spread", 0.005)
        same_bank_fee_rate = fee_configs.get("same_bank_fee_rate", 0.0)

        # 创建 TransactionAgent（子图级生成器）
        self.transaction_agent = TransactionAgent(
            name="transaction_agent",
            manager_agent=manager_agent,
            model_config_name="transaction_manager",  # 使用交易管理器配置
            random_seed=self.random_seed,
            default_currency=default_currency,
            payment_currencies=currencies,
            receiving_currencies=currencies,
            cross_currency_ratio=cross_currency_ratio,
            exchange_rates=exchange_rates,
            same_currency_fee_rate=same_currency_fee_rate,
            cross_currency_fee_rate=cross_currency_fee_rate,
            exchange_rate_spread=exchange_rate_spread,
            same_bank_fee_rate=same_bank_fee_rate,
        )

    def initialize(self):
        """初始化环境（对应 SocialEnvironment.initialize）- 同步版本"""
        # 调用 manager 的 load_data 加载数据
        response = self.manager_agent(
            Msg(
                "user",
                content="load_data",
                kwargs={
                    "start_time": self.time_configs["start_time"],
                    "end_time": self.time_configs["end_time"],
                },
                func="load_data"
            )
        )

        # 检查响应
        if response.content.startswith("Error"):
            raise RuntimeError(f"Failed to load data: {response.content}")

        if self.manager_agent.manager is None:
            raise RuntimeError("Manager was not initialized after load_data")

        # 从manager恢复cur_time（如果有state.json）
        if hasattr(self.manager_agent.manager, 'cur_time') and self.manager_agent.manager.cur_time:
            self.time_configs["cur_time"] = self.manager_agent.manager.cur_time
            print(f"Resumed from {self.time_configs['cur_time']}, total transactions: {self.manager_agent.manager.total_txn_generated}")

        # 加载 scenario 缓存（如果存在）
        self._load_scenario_cache()

        print("Transaction environment initialized")

    async def initialize_async(self):
        """初始化环境（对应 SocialEnvironment.initialize）- 异步版本"""
        # 调用 manager 的 load_data_async 加载数据
        message = Msg(
            "user",
            content="load_data",
            kwargs={
                "start_time": self.time_configs["start_time"],
                "end_time": self.time_configs["end_time"],
            },
            func="load_data"
        )

        response = await self.manager_agent._handle_load_data_async(message)

        # 检查响应
        if response.content.startswith("Error"):
            raise RuntimeError(f"Failed to load data: {response.content}")

        if self.manager_agent.manager is None:
            raise RuntimeError("Manager was not initialized after load_data")

        # 从manager恢复cur_time（如果有state.json）
        if hasattr(self.manager_agent.manager, 'cur_time') and self.manager_agent.manager.cur_time:
            self.time_configs["cur_time"] = self.manager_agent.manager.cur_time
            print(f"Resumed from {self.time_configs['cur_time']}, total transactions: {self.manager_agent.manager.total_txn_generated}")

        # 加载 scenario 缓存（如果存在）
        self._load_scenario_cache()

        print("Transaction environment initialized")

    def step(self):
        """执行一个时间步（生成该窗口内的交易）- 同步版本"""
        window_start = self.time_configs["cur_time"]
        window_end = window_start + timedelta(minutes=self.time_configs["window_size_minutes"])

        # 确保不超过结束时间
        if window_end > self.time_configs["end_time"]:
            window_end = self.time_configs["end_time"]

        # 检查是否是新的一天（每天第一个窗口）
        cur_date = window_start.date()
        is_new_day = (window_start.hour == 0 and window_start.minute == 0)

        # 如果是新的一天，调用 LLM 生成场景
        scenario_json = None
        if is_new_day:
            # 构建 global_stats
            manager = self.manager_agent.manager
            total_txn_so_far = len(manager.transactions) if manager and hasattr(manager, 'transactions') else 0
            target_txn = self.transaction_configs["target_txn_count"]
            remaining_txn = max(0, target_txn - total_txn_so_far)

            global_stats = {
                "cur_date": str(cur_date),
                "total_txn_so_far": total_txn_so_far,
                "remaining_txn": remaining_txn,
                "target_txn_count": target_txn,
            }

            # 调用 LLM 生成场景
            scenario_json = self.llm_scenario_day(cur_date, global_stats)
        else:
            # 使用缓存的场景（如果存在）
            scenario_json = self.daily_scenario_cache.get(cur_date)

        # 计算窗口内需要生成的交易数（基于 tod_p 分布）
        total_txn_target = self.transaction_configs["target_txn_count"]

        # 计算当前窗口的时间权重（基于所有账户的 tod_p 平均分布）
        window_weight = self._calculate_window_weight(window_start, window_end)

        # 使用累积器避免整数截断误差
        if not hasattr(self, '_txn_accumulator'):
            self._txn_accumulator = 0.0
            self._total_weight = self._calculate_total_weight()

        # 计算本窗口应生成的交易数（基于权重分配）
        base_txn_count = total_txn_target * window_weight / self._total_weight

        # 应用 volume_multiplier（如果有场景）- 在权重分配之后应用
        if scenario_json:
            base_txn_count *= scenario_json["volume_multiplier"]

        self._txn_accumulator += base_txn_count
        txn_count_in_window = int(self._txn_accumulator)
        self._txn_accumulator -= txn_count_in_window

        # 每天第一个窗口开始时，保存前一天的账户到CSV
        # 这样确保保存的账户都已经参与了前一天的交易
        if window_start.hour == 0 and window_start.minute == 0:
            prev_day = window_start - timedelta(days=1)
            print(f"\n{'='*70}")
            print(f"保存前一天账户表: {prev_day.date()}")
            self.manager_agent.manager._save_account_tables()
            print(f"{'='*70}\n")

            # 调试信息
            volume_mult = scenario_json.get("volume_multiplier", 1.0) if scenario_json else 1.0
            print(f"[DEBUG] 窗口权重: {window_weight:.6f}, 总权重: {self._total_weight:.6f}, volume_multiplier: {volume_mult:.2f}, 应生成: {txn_count_in_window} 笔")

        if txn_count_in_window == 0:
            # 最后一个窗口可能没有交易
            self.time_configs["cur_time"] = window_end
            return

        print(f"Generating {txn_count_in_window} transactions for {window_start} - {window_end}")

        # 1. 调用 manager.allocate_budget 按 geo 分配交易预算
        allocation = self.manager_agent.allocate_budget(
            window_start=window_start,
            window_end=window_end,
            total_events=txn_count_in_window
        )

        # 2. 应用 geo_multipliers（如果有场景）
        if scenario_json and scenario_json.get("geo_multipliers"):
            geo_multipliers = scenario_json["geo_multipliers"]
            for geo_id in allocation:
                geo_key = str(geo_id)
                if geo_key in geo_multipliers:
                    multiplier = geo_multipliers[geo_key]
                    allocation[geo_id] = round(allocation[geo_id] * multiplier)

        # 3. 对每个 geo 调用 TransactionAgent.generate_batch 生成交易提案
        all_proposals = []
        for geo_id, n_events in allocation.items():
            if n_events <= 0:
                continue

            proposals = self.transaction_agent.generate_batch(
                geo_id=geo_id,
                window_start=window_start,
                window_end=window_end,
                n_events=n_events
            )
            all_proposals.extend(proposals)

        # 3. 调用 manager.apply_events 落地交易
        if all_proposals:
            self.manager_agent.apply_events(all_proposals)

        # 4. 定期更新 top merchants（每月第一天）
        if self.time_configs["cur_time"].day == 1 and window_end.day != 1:
            self.manager_agent.update_top_merchants()

        # 5. 每天结束时保存 state.json（方便检查、暂停、恢复）
        if window_end.hour == 0 and window_end.minute == 0 and window_start.hour != 0:
            print(f"\n{'='*70}")
            print(f"每日保存: {window_start.date()}")
            self.manager_agent.manager.save_infos(
                cur_time=window_end,
                start_time=self.time_configs["start_time"],
                force=True
            )
            print(f"{'='*70}\n")

        # 6. 推进时间
        self.time_configs["cur_time"] = window_end

    async def step_async(self, max_parallel: int = 5):
        """
        执行一个时间步（生成该窗口内的交易）- 异步并行版本

        Args:
            max_parallel: 最大并行数（控制同时执行的 geo 数量）
        """
        window_start = self.time_configs["cur_time"]
        window_end = window_start + timedelta(minutes=self.time_configs["window_size_minutes"])

        # 确保不超过结束时间
        if window_end > self.time_configs["end_time"]:
            window_end = self.time_configs["end_time"]

        # 检查是否是新的一天（每天第一个窗口）
        cur_date = window_start.date()
        is_new_day = (window_start.hour == 0 and window_start.minute == 0)

        # 如果是新的一天，调用 LLM 生成场景
        scenario_json = None
        if is_new_day:
            # 构建 global_stats
            manager = self.manager_agent.manager
            total_txn_so_far = len(manager.transactions) if manager and hasattr(manager, 'transactions') else 0
            target_txn = self.transaction_configs["target_txn_count"]
            remaining_txn = max(0, target_txn - total_txn_so_far)

            global_stats = {
                "cur_date": str(cur_date),
                "total_txn_so_far": total_txn_so_far,
                "remaining_txn": remaining_txn,
                "target_txn_count": target_txn,
            }

            # 调用 LLM 生成场景
            scenario_json = self.llm_scenario_day(cur_date, global_stats)
        else:
            # 使用缓存的场景（如果存在）
            scenario_json = self.daily_scenario_cache.get(cur_date)

        # 计算窗口内需要生成的交易数（基于 tod_p 分布）
        total_txn_target = self.transaction_configs["target_txn_count"]

        # 计算当前窗口的时间权重（基于所有账户的 tod_p 平均分布）
        window_weight = self._calculate_window_weight(window_start, window_end)

        # 使用累积器避免整数截断误差
        if not hasattr(self, '_txn_accumulator'):
            self._txn_accumulator = 0.0
            self._total_weight = self._calculate_total_weight()

        # 计算本窗口应生成的交易数（基于权重分配）
        base_txn_count = total_txn_target * window_weight / self._total_weight

        # 应用 volume_multiplier（如果有场景）- 在权重分配之后应用
        if scenario_json:
            base_txn_count *= scenario_json["volume_multiplier"]

        self._txn_accumulator += base_txn_count
        txn_count_in_window = int(self._txn_accumulator)
        self._txn_accumulator -= txn_count_in_window

        # 每天第一个窗口开始时，保存前一天的账户到CSV
        # 这样确保保存的账户都已经参与了前一天的交易
        if window_start.hour == 0 and window_start.minute == 0:
            prev_day = window_start - timedelta(days=1)
            print(f"\n{'='*70}")
            print(f"保存前一天账户表: {prev_day.date()}")
            self.manager_agent.manager._save_account_tables()
            print(f"{'='*70}\n")

            # 调试信息
            volume_mult = scenario_json.get("volume_multiplier", 1.0) if scenario_json else 1.0
            print(f"[DEBUG] 窗口权重: {window_weight:.6f}, 总权重: {self._total_weight:.6f}, volume_multiplier: {volume_mult:.2f}, 应生成: {txn_count_in_window} 笔")

        if txn_count_in_window == 0:
            # 最后一个窗口可能没有交易
            self.time_configs["cur_time"] = window_end
            return

        print(f"Generating {txn_count_in_window} transactions for {window_start} - {window_end} (async mode)")

        # 1. 调用 manager.allocate_budget 按 geo 分配交易预算（串行，很快）
        allocation = self.manager_agent.allocate_budget(
            window_start=window_start,
            window_end=window_end,
            total_events=txn_count_in_window
        )

        # 2. 应用 geo_multipliers（如果有场景）
        if scenario_json and scenario_json.get("geo_multipliers"):
            geo_multipliers = scenario_json["geo_multipliers"]
            for geo_id in allocation:
                geo_key = str(geo_id)
                if geo_key in geo_multipliers:
                    multiplier = geo_multipliers[geo_key]
                    allocation[geo_id] = round(allocation[geo_id] * multiplier)

        # 2. 并行调用 TransactionAgent.generate_batch 生成交易提案
        all_proposals = await self._generate_batch_parallel(
            allocation=allocation,
            window_start=window_start,
            window_end=window_end,
            max_parallel=max_parallel
        )

        # 3. 调用 manager.apply_events 落地交易（串行，必须保证顺序）
        if all_proposals:
            self.manager_agent.apply_events(all_proposals)

        # 4. 定期更新 top merchants（每月第一天）
        if self.time_configs["cur_time"].day == 1 and window_end.day != 1:
            self.manager_agent.update_top_merchants()

        # 5. 每日账户淘汰和新增（每天最后一个时间窗口）
        # 注意：新账户在当天结束时生成，但要等到第二天开始时才保存到CSV
        if window_end.hour == 0 and window_end.minute == 0 and window_start.hour != 0:
            await self._daily_account_churn_async()

            # 只保存 state.json（账户表在第二天开始时保存）
            print(f"\n{'='*70}")
            print(f"每日保存 state.json: {window_start.date()}")
            self.manager_agent.manager.save_infos(
                cur_time=window_end,
                start_time=self.time_configs["start_time"],
                force=True
            )
            print(f"{'='*70}\n")

        # 6. 推进时间
        self.time_configs["cur_time"] = window_end

    async def _generate_batch_parallel(
        self,
        allocation: Dict[int, int],
        window_start: datetime,
        window_end: datetime,
        max_parallel: int = 5
    ) -> List[Dict[str, Any]]:
        """
        并行生成多个 geo 的交易批次

        Args:
            allocation: geo_id -> n_events 的分配字典
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            max_parallel: 最大并行数

        Returns:
            所有交易提案的列表
        """
        # 过滤掉 n_events <= 0 的 geo
        valid_geos = [(geo_id, n_events) for geo_id, n_events in allocation.items() if n_events > 0]

        if not valid_geos:
            return []

        # 创建异步任务
        async def generate_for_geo(geo_id: int, n_events: int):
            """为单个 geo 生成交易（在线程池中执行同步函数）"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,  # 使用默认线程池
                self.transaction_agent.generate_batch,
                geo_id,
                window_start,
                window_end,
                n_events
            )

        # 分批并行执行（每批 max_parallel 个）
        all_proposals = []
        for batch_start in range(0, len(valid_geos), max_parallel):
            batch = valid_geos[batch_start:batch_start + max_parallel]

            # 创建当前批次的任务
            tasks = [generate_for_geo(geo_id, n_events) for geo_id, n_events in batch]

            # 并行执行当前批次
            batch_results = await asyncio.gather(*tasks)

            # 收集结果
            for proposals in batch_results:
                if proposals:
                    all_proposals.extend(proposals)

        return all_proposals

    async def _daily_account_churn_async(self, churn_ratio: float = 0.01):
        """
        每日账户淘汰和新增（异步版本）

        Args:
            churn_ratio: 淘汰和新增的比例（默认10%）
        """
        manager = self.manager_agent.manager
        if manager is None:
            return

        print(f"\n[Daily Account Churn] Starting daily account churn at {self.time_configs['cur_time']}")

        # 1. 删除交易数量最少的账户
        removed_persons, removed_merchants = manager.remove_inactive_accounts(removal_ratio=churn_ratio)

        # 2. 生成新账户（数量与删除的相同）
        num_new_persons = len(removed_persons)
        num_new_merchants = len(removed_merchants)

        if num_new_persons > 0 or num_new_merchants > 0:
            new_persons, new_merchants = await manager.add_new_accounts_async(
                num_persons=num_new_persons,
                num_merchants=num_new_merchants,
                max_parallel=5
            )

            # 3. 重建采样加速结构
            manager._build_sender_cdf()

            print(f"[Daily Account Churn] Completed: removed {len(removed_persons)}+{len(removed_merchants)}, added {len(new_persons)}+{len(new_merchants)}")
            print(f"[Daily Account Churn] Current total: {len(manager.person_ids)} persons, {len(manager.merchant_ids)} merchants\n")

    def save(self, start_time=None):
        """保存环境状态（对应 SocialEnvironment.save）"""
        if start_time is None:
            start_time = self.time_configs["start_time"]

        # 调用 manager_agent.save_infos
        self.manager_agent.save_infos(
            cur_time=self.time_configs["cur_time"],
            start_time=start_time,
            force=True
        )

        print(f"Saved transaction environment state at {self.time_configs['cur_time']}")

    def is_done(self) -> bool:
        """检查是否完成（对应 SocialEnvironment.is_done）"""
        return self.time_configs["cur_time"] >= self.time_configs["end_time"]

    def reset(self) -> None:
        """重置环境（对应 SocialEnvironment.reset）"""
        self.time_configs["cur_time"] = self.time_configs["start_time"]

    def _calculate_window_weight(self, window_start: datetime, window_end: datetime) -> float:
        """
        计算窗口的时间权重（基于所有账户的 tod_p 平均分布）

        Args:
            window_start: 窗口开始时间
            window_end: 窗口结束时间

        Returns:
            窗口权重（0-1之间的值）
        """
        # 获取所有个人账户的平均 tod_p 分布
        if not hasattr(self.manager_agent.manager, 'account_map'):
            return 1.0

        accounts = self.manager_agent.manager.account_map
        tod_p_sum = [0.0] * 8
        person_count = 0

        for acc_id, acc in accounts.items():
            if acc.get('account_type') == 'person':
                tod_p = acc.get('tags', {}).get('tod_p', [])
                if len(tod_p) == 8:
                    person_count += 1
                    for i in range(8):
                        tod_p_sum[i] += tod_p[i]

        if person_count == 0:
            return 1.0

        # 计算平均 tod_p
        tod_p_avg = [s / person_count for s in tod_p_sum]

        # 计算窗口覆盖的时间段权重
        # tod_p[i] 表示该3小时时段占全天交易的比例
        # 窗口权重 = 窗口时长（小时）* 该时段的小时权重
        window_duration_hours = (window_end - window_start).total_seconds() / 3600

        # 获取窗口所在的时段
        window_start_hour = window_start.hour
        segment = window_start_hour // 3

        # 该时段的小时权重 = tod_p[segment] / 3（因为每段是3小时）
        hour_weight = tod_p_avg[segment] / 3.0

        # 窗口权重 = 窗口时长 * 小时权重
        window_weight = window_duration_hours * hour_weight

        return window_weight

    def _calculate_total_weight(self) -> float:
        """
        计算整个时间跨度的总权重（用于归一化）

        Returns:
            总权重
        """
        total_weight = 0.0
        current_time = self.time_configs["start_time"]
        end_time = self.time_configs["end_time"]
        window_size = timedelta(minutes=self.time_configs["window_size_minutes"])

        while current_time < end_time:
            window_end = min(current_time + window_size, end_time)
            total_weight += self._calculate_window_weight(current_time, window_end)
            current_time = window_end

        return total_weight

    def _setup_llm_logger(self, log_dir: str):
        """
        设置 LLM 日志记录器

        Args:
            log_dir: 日志输出目录
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件路径
        log_file = os.path.join(log_dir, "llm_scenario_generator.log")

        # 创建 logger
        self.llm_logger = logging.getLogger(f"scenario_generator_{id(self)}")
        self.llm_logger.setLevel(logging.INFO)

        # 避免重复添加 handler
        if not self.llm_logger.handlers:
            # 创建文件 handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)

            # 添加 handler
            self.llm_logger.addHandler(file_handler)

    def llm_scenario_day(self, cur_date: date, global_stats: dict) -> dict:
        """
        每日一次的 LLM 场景生成函数（带重试机制）

        调用本地部署的 LLM 生成当日场景参数，包括：
        - volume_multiplier: 当日总体交易量倍率 [0.5, 1.5]
        - geo_multipliers: geo 级别倍率 [0.5, 2.0]
        - promotion_theme: 促销主题/事件主题

        Args:
            cur_date: 当前日期
            global_stats: 全局统计信息（用于 prompt）

        Returns:
            scenario_json: 场景参数字典
        """
        # 记录开始日志
        self.llm_logger.info(f"=== Starting scenario generation for {cur_date} ===")
        self.llm_logger.info(f"Global stats: {json.dumps(global_stats, indent=2, default=str)}")

        # 检查缓存
        if cur_date in self.daily_scenario_cache:
            self.llm_logger.info(f"Using cached scenario for {cur_date}")
            return self.daily_scenario_cache[cur_date]

        # 默认场景（降级方案）
        default_scenario = {
            "volume_multiplier": 1.0,
            "geo_multipliers": {},
            "promotion_theme": "normal"
        }

        # 重试配置
        max_retries = 3

        # 加载 LLM 模型（在重试循环外，只加载一次）
        try:
            self.llm_logger.info(f"Loading model: {self.model_config_name}")
            model = load_model_by_config_name(self.model_config_name)
        except Exception as e:
            self.llm_logger.error(f"Failed to load model: {e}", exc_info=True)
            self.daily_scenario_cache[cur_date] = default_scenario
            return default_scenario

        # 使用 prompt 模板构建 prompt（在重试循环外，只构建一次）
        system_prompt, user_prompt = build_daily_scenario_prompt(global_stats)
        self.llm_logger.info(f"System prompt: {system_prompt}")
        self.llm_logger.info(f"User prompt: {user_prompt}")

        # 重试循环
        for retry_count in range(1, max_retries + 1):
            try:
                self.llm_logger.info(f"Attempt {retry_count}/{max_retries}: Calling LLM...")

                # 调用 LLM
                response = model(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                self.llm_logger.info(f"Attempt {retry_count}: LLM call completed")

                # 记录完整的响应对象信息
                self.llm_logger.info(f"Attempt {retry_count}: Response object type: {type(response)}")
                self.llm_logger.info(f"Attempt {retry_count}: Response object repr: {repr(response)}")

                # 记录所有可能包含响应内容的属性
                if hasattr(response, 'raw'):
                    self.llm_logger.info(f"Attempt {retry_count}: Response.raw = {response.raw}")
                if hasattr(response, 'text'):
                    self.llm_logger.info(f"Attempt {retry_count}: Response.text = {response.text}")
                if hasattr(response, 'parsed'):
                    self.llm_logger.info(f"Attempt {retry_count}: Response.parsed = {response.parsed}")
                if hasattr(response, 'content'):
                    self.llm_logger.info(f"Attempt {retry_count}: Response.content = {response.content}")

                # 解析响应 - 按优先级尝试不同属性
                response_text = None

                # 优先尝试从 response.raw 字典中提取 content
                if hasattr(response, 'raw') and isinstance(response.raw, dict):
                    try:
                        content = response.raw.get('choices', [{}])[0].get('message', {}).get('content', '')
                        if content:
                            response_text = content
                            self.llm_logger.info(f"Attempt {retry_count}: Using response.raw['choices'][0]['message']['content']")
                        else:
                            self.llm_logger.warning(f"Attempt {retry_count}: response.raw['choices'][0]['message']['content'] is empty")
                    except (KeyError, IndexError, TypeError) as e:
                        self.llm_logger.warning(f"Attempt {retry_count}: Failed to extract from response.raw dict: {e}")

                # 如果上面没有提取到，尝试其他属性
                if not response_text:
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text
                        self.llm_logger.info(f"Attempt {retry_count}: Using response.text")
                    elif hasattr(response, 'parsed') and response.parsed:
                        response_text = str(response.parsed)
                        self.llm_logger.info(f"Attempt {retry_count}: Using response.parsed")
                    elif hasattr(response, 'content') and response.content:
                        response_text = response.content
                        self.llm_logger.info(f"Attempt {retry_count}: Using response.content")
                    elif hasattr(response, 'raw') and response.raw:
                        response_text = str(response.raw)
                        self.llm_logger.info(f"Attempt {retry_count}: Using str(response.raw)")
                    else:
                        response_text = str(response)
                        self.llm_logger.info(f"Attempt {retry_count}: Using str(response)")

                self.llm_logger.info(f"Attempt {retry_count}: Final response_text = {response_text}")

                # 尝试提取 JSON（处理可能的代码块包装）
                response_text = response_text.strip()

                # 检查响应是否为空
                if not response_text:
                    self.llm_logger.warning(f"Attempt {retry_count}: LLM returned empty response")
                    raise ValueError("LLM returned empty response")

                if response_text.startswith("```"):
                    # 移除代码块标记
                    lines = response_text.split("\n")
                    response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                    self.llm_logger.info(f"Attempt {retry_count}: Cleaned response: {response_text}")

                # 再次检查清理后的响应是否为空
                if not response_text:
                    self.llm_logger.warning(f"Attempt {retry_count}: LLM response is empty after cleaning")
                    raise ValueError("LLM response is empty after cleaning")

                scenario_json = json.loads(response_text)
                self.llm_logger.info(f"Attempt {retry_count}: Parsed JSON: {json.dumps(scenario_json, indent=2)}")

                # 验证和约束字段
                volume_multiplier = float(scenario_json.get("volume_multiplier", 1.0))
                volume_multiplier = max(0.5, min(1.5, volume_multiplier))

                geo_multipliers = scenario_json.get("geo_multipliers", {})
                geo_multipliers_clamped = {}
                for geo_id, multiplier in geo_multipliers.items():
                    multiplier = float(multiplier)
                    multiplier = max(0.5, min(2.0, multiplier))
                    geo_multipliers_clamped[str(geo_id)] = multiplier

                promotion_theme = scenario_json.get("promotion_theme", "normal")
                if promotion_theme not in ["payday", "holiday_sale", "travel_season", "normal"]:
                    promotion_theme = "normal"

                # 构建最终场景
                final_scenario = {
                    "volume_multiplier": volume_multiplier,
                    "geo_multipliers": geo_multipliers_clamped,
                    "promotion_theme": promotion_theme
                }

                # 缓存结果
                self.daily_scenario_cache[cur_date] = final_scenario

                # 立即保存缓存到文件
                self._save_scenario_cache()

                # 记录成功日志
                self.llm_logger.info(f"Final scenario: {json.dumps(final_scenario, indent=2)}")
                self.llm_logger.info(f"Successfully generated scenario for {cur_date}: volume={volume_multiplier:.2f}, theme={promotion_theme}, geos={len(geo_multipliers_clamped)}")
                print(f"[LLM Scenario] {cur_date}: volume={volume_multiplier:.2f}, theme={promotion_theme}, geos={len(geo_multipliers_clamped)}")

                return final_scenario

            except Exception as e:
                # 记录重试错误
                self.llm_logger.warning(f"Attempt {retry_count}/{max_retries} failed: {e}")
                
                # 如果还有重试机会，继续
                if retry_count < max_retries:
                    self.llm_logger.info(f"Retrying... ({retry_count + 1}/{max_retries})")
                    continue
                else:
                    # 所有重试都失败，记录最终错误
                    self.llm_logger.error(f"All {max_retries} attempts failed for {cur_date}", exc_info=True)
                    self.llm_logger.info(f"Using default scenario: {json.dumps(default_scenario, indent=2)}")
                    print(f"[LLM Scenario] Failed to generate scenario for {cur_date} after {max_retries} attempts")
                    print(f"[LLM Scenario] Using default scenario")

                    # 缓存默认场景以避免重复失败
                    self.daily_scenario_cache[cur_date] = default_scenario

                    # 保存缓存到文件
                    self._save_scenario_cache()

                    return default_scenario

    def _load_scenario_cache(self):
        """从文件加载 scenario 缓存"""
        cache_file = os.path.join(self.output_dir, "scenario_cache.json")

        if not os.path.exists(cache_file):
            self.llm_logger.info("No scenario cache file found, starting fresh")
            return

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # 将字符串日期转换回 date 对象
            self.daily_scenario_cache = {}
            for date_str, scenario in cache_data.items():
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                self.daily_scenario_cache[date_obj] = scenario

            self.llm_logger.info(f"Loaded {len(self.daily_scenario_cache)} cached scenarios")
            print(f"✓ 加载了 {len(self.daily_scenario_cache)} 个缓存的 scenario")
        except Exception as e:
            self.llm_logger.error(f"Failed to load scenario cache: {e}", exc_info=True)
            print(f"✗ 加载 scenario 缓存失败: {e}")

    def _save_scenario_cache(self):
        """保存 scenario 缓存到文件"""
        cache_file = os.path.join(self.output_dir, "scenario_cache.json")

        try:
            # 将 date 对象转换为字符串
            cache_data = {}
            for date_obj, scenario in self.daily_scenario_cache.items():
                date_str = date_obj.strftime("%Y-%m-%d")
                cache_data[date_str] = scenario

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.llm_logger.info(f"Saved {len(cache_data)} scenarios to cache")
        except Exception as e:
            self.llm_logger.error(f"Failed to save scenario cache: {e}", exc_info=True)
