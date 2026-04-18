"""
TransactionAgent: 子图级交易生成器

职责：
- 按 geo_id 和预算 n_events 编排生成交易提案
- 不维护全局状态，只负责生成交易 dict 列表
- 所有状态查询、采样、落地均通过 manager_agent 或 manager 完成
"""

import random
import json
import os
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, date

from agentscope.agents import AgentBase
from agentscope.message import Msg

from LLMGraph.agent import agent_registry
from LLMGraph.prompt.transaction import build_transaction_planning_prompt


@agent_registry.register("transaction_agent")
class TransactionAgent(AgentBase):
    """子图级交易生成器（不做每账户一个 agent）"""

    def __init__(
        self,
        name: str,
        manager_agent=None,
        manager=None,
        model_config_name: Optional[str] = "default",
        default_currency: str = "USD",
        payment_currencies: Optional[List[str]] = None,
        receiving_currencies: Optional[List[str]] = None,
        cross_currency_ratio: float = 0.05,
        exchange_rates: Optional[Dict[str, float]] = None,
        same_currency_fee_rate: float = 0.001,
        cross_currency_fee_rate: float = 0.015,
        exchange_rate_spread: float = 0.005,
        same_bank_fee_rate: float = 0.0,  # 同银行内转账手续费率
        sender_sampling: str = "weighted",
        to_merchant_only: bool = True,
        random_seed: int = 42,
        **kwargs
    ):
        """
        初始化 TransactionAgent

        Args:
            name: agent 名称
            manager_agent: TransactionManagerAgent 实例（优先）
            manager: TransactionManager 实例（备选）
            model_config_name: 模型配置名称（用于 LLM 规划）
            default_currency: 默认货币
            payment_currencies: 支付货币列表
            receiving_currencies: 接收货币列表
            cross_currency_ratio: 跨币种交易占比（默认 0.05 = 5%）
            exchange_rates: 汇率字典 {currency: rate_to_USD}
            same_currency_fee_rate: 同币种交易手续费率（默认 0.001 = 0.1%）
            cross_currency_fee_rate: 跨币种交易手续费率（默认 0.015 = 1.5%）
            exchange_rate_spread: 汇率价差（默认 0.005 = 0.5%）
            sender_sampling: sender 采样方式（"weighted" 或 "uniform"）
            to_merchant_only: 是否只生成 P2M 交易
            random_seed: 随机种子
            **kwargs: 其他 AgentBase 参数
        """
        super().__init__(name=name, model_config_name=model_config_name, **kwargs)

        self.manager_agent = manager_agent
        self.manager = manager

        # 优先使用 manager_agent
        if self.manager_agent is not None and hasattr(self.manager_agent, "manager"):
            self.manager = self.manager_agent.manager

        # 货币配置
        self.default_currency = default_currency
        self.payment_currencies = payment_currencies or [default_currency]
        self.receiving_currencies = receiving_currencies or [default_currency]
        self.cross_currency_ratio = cross_currency_ratio

        # 汇率配置
        self.exchange_rates = exchange_rates or {default_currency: 1.0}

        # 手续费配置
        self.same_currency_fee_rate = same_currency_fee_rate
        self.cross_currency_fee_rate = cross_currency_fee_rate
        self.exchange_rate_spread = exchange_rate_spread
        self.same_bank_fee_rate = same_bank_fee_rate

        self.sender_sampling = sender_sampling
        self.to_merchant_only = to_merchant_only

        # 支付方式集合（用于 plan_json 验证）
        # 中国大陆常用支付方式：
        # - Mobile: 移动支付（微信/支付宝），主要用于日常消费、扫码支付
        # - Card: 银行卡支付（借记卡/信用卡），用于刷卡消费
        # - Transfer: 银行转账，用于大额转账、工资发放等
        # - Cash: 现金支付
        self.payment_formats = ["Mobile", "Card", "Transfer", "Cash"]

        # 本地 RNG（用于备用采样）
        self._rng = random.Random(random_seed)

    def reply(self, message: Msg) -> Msg:
        """
        处理消息并返回响应

        支持的命令：
        - "generate_batch": 生成交易批次
        - "ping": 健康检查

        Args:
            message: 输入消息

        Returns:
            响应消息
        """
        content = message.content

        try:
            if content == "generate_batch":
                geo_id = message.get("geo_id")
                window_start = message.get("window_start")
                window_end = message.get("window_end")
                n_events = message.get("n_events", 0)

                transactions = self.generate_batch(
                    geo_id=geo_id,
                    window_start=window_start,
                    window_end=window_end,
                    n_events=n_events
                )

                return Msg(
                    name=self.name,
                    role="assistant",
                    content=transactions
                )
            elif content == "ping":
                return Msg(
                    name=self.name,
                    role="assistant",
                    content="ok"
                )
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
                content=f"Error: {str(e)}"
            )

    def generate_batch(
        self,
        geo_id: int,
        window_start: datetime,
        window_end: datetime,
        n_events: int
    ) -> List[Dict[str, Any]]:
        """
        生成一个 geo 内的交易批次（每 geo 每轮调用一次 LLM 规划）

        Args:
            geo_id: geo ID
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            n_events: 交易数量

        Returns:
            交易字典列表（10字段齐全）
        """
        if n_events <= 0:
            return []

        # 获取 geo 内的个人账户和商户账户
        if self.manager_agent is not None:
            persons = self.manager_agent.get_geo_person_ids(geo_id)
            merchants = self.manager_agent.get_geo_merchant_ids(geo_id)
        elif self.manager is not None:
            persons = self.manager.geo_persons.get(geo_id, [])
            merchants = self.manager.geo_merchants.get(geo_id, [])
        else:
            return []

        if not persons:
            return []

        # 构造 geo_summary 用于 LLM 规划
        geo_summary = {
            "geo_id": geo_id,
            "n_persons": len(persons),
            "n_merchants": len(merchants),
            "target_txn_count": n_events,
        }

        # 添加 top_merchants（可选）
        if self.manager is not None and hasattr(self.manager, 'top_merchants'):
            top_merchants = getattr(self.manager, 'top_merchants', [])
            if top_merchants:
                geo_summary["top_merchants"] = top_merchants[:10]  # 取前10个

        # 添加 recent_stats（可选）
        # 这里可以从 manager 获取历史统计，暂时使用默认值
        geo_summary["recent_stats"] = {
            "p2m_ratio": 0.75,
            "cross_bank_ratio": 0.1,
            "format_distribution": {fmt: 0.2 for fmt in self.payment_formats}
        }

        # 调用 LLM 规划（每 geo 每轮 1 次）
        plan_json = self.llm_plan_round(geo_id, window_start, window_end, geo_summary)

        # 采样 senders
        senders = self._sample_senders(persons, n_events)

        # 生成交易（使用 plan_json 控制生成分布）
        transactions = []
        for sender_id in senders:
            txn = self._generate_transaction(
                sender_id=sender_id,
                geo_id=geo_id,
                window_start=window_start,
                window_end=window_end,
                plan_json=plan_json  # 传递 LLM 规划
            )
            if txn is not None:
                transactions.append(txn)

        return transactions

    def llm_plan_round(
        self,
        geo_id: int,
        window_start: datetime,
        window_end: datetime,
        geo_summary: dict
    ) -> dict:
        """
        使用 LLM 规划本轮交易生成策略

        Args:
            geo_id: geo ID
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            geo_summary: geo 摘要信息（包含 geo_id, n_persons, n_merchants, top_merchants, recent_stats 等）

        Returns:
            plan_json: LLM 输出的规划 JSON，包含 cross_delta, amount_buckets, payment_format_p, merchant_type_bias
        """
        # 构造 prompt
        prompt = self._build_planning_prompt(geo_id, window_start, window_end, geo_summary)

        # 调用 LLM（通过 agentscope 的模型接口）
        try:
            # 使用 self.model 调用 LLM
            response = self.model(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )

            # 解析响应
            response_text = response.text if hasattr(response, 'text') else str(response)

            # 提取 JSON
            plan_json = self._extract_json_from_response(response_text)

            # 验证和归一化 plan_json
            plan_json = self._validate_and_normalize_plan(plan_json)

            return plan_json

        except Exception as e:
            # 如果 LLM 调用失败，返回默认规划
            print(f"Warning: LLM planning failed for geo {geo_id}: {e}")
            return self._get_default_plan()

    def _build_planning_prompt(
        self,
        geo_id: int,
        window_start: datetime,
        window_end: datetime,
        geo_summary: dict
    ) -> str:
        """
        构造 LLM 规划 prompt（使用统一的 prompt 模板）

        Args:
            geo_id: geo ID
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            geo_summary: geo 摘要信息

        Returns:
            prompt 字符串
        """
        return build_transaction_planning_prompt(
            geo_id=geo_id,
            window_start=window_start,
            window_end=window_end,
            geo_summary=geo_summary
        )

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        从 LLM 响应中提取 JSON

        Args:
            response_text: LLM 响应文本

        Returns:
            解析后的 JSON 字典
        """
        try:
            # 尝试直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # 尝试查找第一个 { 到最后一个 }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(response_text[start_idx:end_idx+1])
                except json.JSONDecodeError:
                    pass

            # 如果都失败，返回默认规划
            return self._get_default_plan()

    def _validate_and_normalize_plan(self, plan_json: dict) -> dict:
        """
        验证和归一化 plan_json

        Args:
            plan_json: 原始 plan_json

        Returns:
            验证和归一化后的 plan_json
        """
        # 验证 cross_delta
        cross_delta = plan_json.get("cross_delta", 0.0)
        cross_delta = max(-0.05, min(0.05, cross_delta))  # clamp 到 [-0.05, 0.05]

        # 验证和归一化 amount_buckets
        amount_buckets = plan_json.get("amount_buckets", [])
        if not amount_buckets or len(amount_buckets) < 2:
            # 使用默认 buckets
            amount_buckets = [
                {"name": "small", "p": 0.5, "mult_range": [0.3, 0.8]},
                {"name": "mid", "p": 0.35, "mult_range": [0.8, 1.5]},
                {"name": "large", "p": 0.15, "mult_range": [1.5, 3.0]}
            ]
        else:
            # 归一化概率
            total_p = sum(bucket.get("p", 0.0) for bucket in amount_buckets)
            if total_p > 1e-6:
                for bucket in amount_buckets:
                    bucket["p"] = bucket.get("p", 0.0) / total_p

        # 验证和归一化 payment_format_p
        payment_format_p = plan_json.get("payment_format_p", {})
        if not payment_format_p:
            # 使用均匀分布
            payment_format_p = {fmt: 1.0 / len(self.payment_formats) for fmt in self.payment_formats}
        else:
            # 只保留有效的 payment_formats
            valid_formats = {k: v for k, v in payment_format_p.items() if k in self.payment_formats}
            if not valid_formats:
                valid_formats = {fmt: 1.0 / len(self.payment_formats) for fmt in self.payment_formats}
            # 归一化概率
            total_p = sum(valid_formats.values())
            if total_p > 1e-6:
                payment_format_p = {k: v / total_p for k, v in valid_formats.items()}
            else:
                payment_format_p = {fmt: 1.0 / len(self.payment_formats) for fmt in self.payment_formats}

        # merchant_type_bias（可选，允许为空）
        merchant_type_bias = plan_json.get("merchant_type_bias", {})

        return {
            "cross_delta": cross_delta,
            "amount_buckets": amount_buckets,
            "payment_format_p": payment_format_p,
            "merchant_type_bias": merchant_type_bias
        }

    def _get_default_plan(self) -> dict:
        """
        获取默认规划（当 LLM 调用失败时使用）

        Returns:
            默认规划 JSON（符合新的 plan_json 契约）
        """
        return {
            "cross_delta": 0.0,
            "amount_buckets": [
                {"name": "small", "p": 0.5, "mult_range": [0.3, 0.8]},
                {"name": "mid", "p": 0.35, "mult_range": [0.8, 1.5]},
                {"name": "large", "p": 0.15, "mult_range": [1.5, 3.0]}
            ],
            "payment_format_p": {fmt: 1.0 / len(self.payment_formats) for fmt in self.payment_formats},
            "merchant_type_bias": {}
        }

    def _sample_senders(self, persons: List[str], n_events: int) -> List[str]:
        """
        采样发起方账户

        Args:
            persons: 个人账户 ID 列表
            n_events: 交易数量

        Returns:
            发起方账户 ID 列表
        """
        if not persons:
            return []

        if self.sender_sampling == "weighted":
            # 加权采样（按 avg_txn_cnt_daily）
            weights = []
            for person_id in persons:
                if self.manager_agent is not None:
                    account = self.manager_agent.get_account(person_id)
                elif self.manager is not None:
                    account = self.manager.account_map.get(person_id)
                else:
                    account = None

                if account is not None:
                    weight = account.get("tags", {}).get("avg_txn_cnt_daily", 1.0)
                else:
                    weight = 1.0

                weights.append(weight)

            # CDF 采样
            total_weight = sum(weights)
            if total_weight <= 0:
                # 降级到均匀采样
                return self._rng.choices(persons, k=n_events)

            cdf = []
            cumsum = 0.0
            for w in weights:
                cumsum += w
                cdf.append(cumsum)

            senders = []
            for _ in range(n_events):
                r = self._rng.uniform(0, total_weight)
                idx = 0
                for i, c in enumerate(cdf):
                    if r <= c:
                        idx = i
                        break
                senders.append(persons[idx])

            return senders
        else:
            # 均匀采样
            return self._rng.choices(persons, k=n_events)

    def _generate_transaction(
        self,
        sender_id: str,
        geo_id: int,
        window_start: datetime,
        window_end: datetime,
        plan_json: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        生成单笔交易（使用 LLM 规划控制生成分布）

        Args:
            sender_id: 发起方账户 ID
            geo_id: geo ID
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            plan_json: LLM 规划 JSON（符合新契约）

        Returns:
            交易字典（10字段），或 None
        """
        # 获取发起方账户
        if self.manager_agent is not None:
            from_acc = self.manager_agent.get_account(sender_id)
        elif self.manager is not None:
            from_acc = self.manager.account_map.get(sender_id)
        else:
            return None

        if from_acc is None:
            return None

        # 从 plan_json 中获取控制参数
        cross_delta = 0.0
        amount_buckets = []
        payment_format_p = {}
        merchant_type_bias = {}

        if plan_json is not None:
            cross_delta = plan_json.get("cross_delta", 0.0)
            amount_buckets = plan_json.get("amount_buckets", [])
            payment_format_p = plan_json.get("payment_format_p", {})
            merchant_type_bias = plan_json.get("merchant_type_bias", {})

        # 决定是否跨 geo（使用 cross_delta 调整基础概率）
        p_cross_base = 0.1  # 基础跨 geo 概率
        p_cross = max(0.0, min(1.0, p_cross_base + cross_delta))
        cross = self._rng.random() < p_cross

        # 选择对手（仍然调用 manager.choose_counterparty，不复制策略）
        to_acc = self._choose_counterparty_with_bias(
            from_acc=from_acc,
            sender_id=sender_id,
            cross=cross,
            merchant_type_bias=merchant_type_bias
        )

        if to_acc is None:
            return None

        to_id = to_acc.get("account_id")

        # 采样时间戳
        timestamp = self._sample_timestamp(
            window_start=window_start,
            window_end=window_end,
            from_acc=from_acc
        )

        # 采样金额（使用 amount_buckets，获取基础金额）
        base_amount_paid, base_amount_received = self._sample_amounts_with_buckets(
            from_acc=from_acc,
            to_acc=to_acc,
            amount_buckets=amount_buckets
        )

        # 采样支付方式（使用 payment_format_p）
        payment_format = self._sample_payment_format_with_plan(payment_format_p)

        # 采样货币（智能采样，支持跨币种交易）
        payment_currency, receiving_currency = self._sample_currencies()

        # 根据货币和汇率计算最终金额
        amount_paid, amount_received = self._calculate_amounts_with_exchange(
            base_amount=base_amount_paid,
            payment_currency=payment_currency,
            receiving_currency=receiving_currency,
            from_bank=from_acc.get("bank"),
            to_bank=to_acc.get("bank")
        )

        # 组装交易字典（10字段）
        transaction = {
            "Timestamp": timestamp.strftime("%Y/%m/%d %H:%M:%S"),
            "From Bank": from_acc.get("bank", "Unknown"),
            "From Account": from_acc.get("bank_account_number", sender_id),
            "To Bank": to_acc.get("bank", "Unknown"),
            "To Account": to_acc.get("bank_account_number", to_id),
            "Amount Received": round(amount_received, 2),
            "Receiving Currency": receiving_currency,
            "Amount Paid": round(amount_paid, 2),
            "Payment Currency": payment_currency,
            "Payment Format": payment_format,
        }

        return transaction

    def _choose_counterparty_with_bias(
        self,
        from_acc: Dict[str, Any],
        sender_id: str,
        cross: bool,
        merchant_type_bias: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        选择对手（支持商户类型偏好）

        Args:
            from_acc: 发起方账户
            sender_id: 发起方账户 ID
            cross: 是否跨 geo
            merchant_type_bias: 商户类型偏好

        Returns:
            对手账户字典，或 None
        """
        # 调用 manager.choose_counterparty
        if self.manager_agent is not None:
            to_acc = self.manager_agent.choose_counterparty(
                from_id=sender_id,
                to_merchant=self.to_merchant_only,
                cross=cross
            )
        elif self.manager is not None:
            to_acc = self.manager.choose_counterparty(from_acc, cross=cross)
        else:
            return None

        # 如果没有商户类型偏好，直接返回
        if not merchant_type_bias or not to_acc:
            return to_acc

        # 尝试重采样以趋向偏好（最多 K 次）
        max_retries = 3
        for _ in range(max_retries):
            merchant_type = to_acc.get("type") or to_acc.get("category")
            if merchant_type and merchant_type in merchant_type_bias:
                # 按偏好概率决定是否接受
                bias_prob = merchant_type_bias.get(merchant_type, 0.0)
                if self._rng.random() < bias_prob:
                    return to_acc

            # 重新采样
            if self.manager_agent is not None:
                to_acc = self.manager_agent.choose_counterparty(
                    from_id=sender_id,
                    to_merchant=self.to_merchant_only,
                    cross=cross
                )
            elif self.manager is not None:
                to_acc = self.manager.choose_counterparty(from_acc, cross=cross)
            else:
                return None

            if not to_acc:
                return None

        return to_acc

    def _sample_timestamp(
        self,
        window_start: datetime,
        window_end: datetime,
        from_acc: Dict[str, Any]
    ) -> datetime:
        """
        采样时间戳

        Args:
            window_start: 窗口开始
            window_end: 窗口结束
            from_acc: 发起方账户

        Returns:
            采样的时间戳
        """
        tod_p = from_acc.get("tags", {}).get("tod_p", [])

        # 优先调用 manager 的 sample_timestamp
        if self.manager_agent is not None:
            return self.manager_agent.sample_timestamp(window_start, window_end, tod_p)
        elif self.manager is not None:
            return self.manager.sample_timestamp(window_start, window_end, tod_p)
        else:
            # 备用实现：8段 tod_p 采样
            return self._fallback_sample_timestamp(window_start, window_end, tod_p)

    def _fallback_sample_timestamp(
        self,
        window_start: datetime,
        window_end: datetime,
        tod_p: List[float]
    ) -> datetime:
        """
        备用时间戳采样（8段 tod_p）

        Args:
            window_start: 窗口开始
            window_end: 窗口结束
            tod_p: 时间分布（8段）

        Returns:
            采样的时间戳
        """
        window_seconds = int((window_end - window_start).total_seconds())

        # 如果没有 tod_p，均匀采样
        if not tod_p or len(tod_p) != 8:
            offset_seconds = self._rng.randint(0, max(0, window_seconds - 1))
            return window_start + timedelta(seconds=offset_seconds)

        # 根据 tod_p 采样时间段（0-7）
        r = self._rng.random()
        cumsum = 0
        segment = 0
        for i, p in enumerate(tod_p):
            cumsum += p
            if r <= cumsum:
                segment = i
                break

        segment_start_hour = segment * 3
        segment_end_hour = min(segment_start_hour + 3, 24)

        window_start_hour = window_start.hour + window_start.minute / 60.0
        window_end_hour = window_end.hour + window_end.minute / 60.0

        # 如果窗口跨天，简化处理：在窗口内均匀采样
        if window_end.date() > window_start.date():
            offset_seconds = self._rng.randint(0, max(0, window_seconds - 1))
            return window_start + timedelta(seconds=offset_seconds)

        # 计算段与窗口的交集
        effective_start_hour = max(segment_start_hour, window_start_hour)
        effective_end_hour = min(segment_end_hour, window_end_hour)

        if effective_start_hour >= effective_end_hour:
            offset_seconds = self._rng.randint(0, max(0, window_seconds - 1))
            return window_start + timedelta(seconds=offset_seconds)

        # 在交集内采样
        effective_start_seconds = int(effective_start_hour * 3600)
        effective_end_seconds = int(effective_end_hour * 3600)
        sampled_seconds = self._rng.randint(effective_start_seconds, max(effective_start_seconds, effective_end_seconds - 1))

        base_date = window_start.date()
        sampled_time = datetime.combine(base_date, datetime.min.time()) + timedelta(seconds=sampled_seconds)

        return sampled_time

    def _sample_amounts(
        self,
        from_acc: Dict[str, Any],
        to_acc: Dict[str, Any],
        multiplier: float = 1.0
    ) -> tuple:
        """
        采样交易金额（支持 LLM 规划的倍数调整）

        Args:
            from_acc: 发起方账户
            to_acc: 对手账户
            multiplier: 金额倍数（来自 LLM 规划）

        Returns:
            (amount_paid, amount_received)
        """
        # 优先调用 manager 的 sample_amounts
        if self.manager_agent is not None:
            amount_paid, amount_received = self.manager_agent.sample_amounts(from_acc, to_acc)
        elif self.manager is not None:
            amount_paid, amount_received = self.manager.sample_amounts(from_acc, to_acc)
        else:
            # 备用实现：按 avg_txn_amt 正态采样
            avg_amt = from_acc.get("tags", {}).get("avg_txn_amt", 50.0)
            std = avg_amt * 0.5
            amount_paid = abs(self._rng.gauss(avg_amt, std))
            amount_paid = max(1.0, amount_paid)

            fee_rate = self._rng.uniform(0.0, 0.02)
            amount_received = amount_paid * (1 - fee_rate)

        # 应用 LLM 规划的倍数
        amount_paid *= multiplier
        amount_received *= multiplier

        return amount_paid, amount_received

    def _sample_amounts_with_buckets(
        self,
        from_acc: Dict[str, Any],
        to_acc: Dict[str, Any],
        amount_buckets: List[Dict[str, Any]]
    ) -> tuple:
        """
        使用 amount_buckets 采样交易金额

        Args:
            from_acc: 发起方账户
            to_acc: 对手账户
            amount_buckets: 金额分段列表

        Returns:
            (amount_paid, amount_received)
        """
        # 如果没有 buckets，使用默认方法
        if not amount_buckets:
            return self._sample_amounts(from_acc, to_acc, multiplier=1.0)

        # 按概率选择 bucket
        r = self._rng.random()
        cumsum = 0.0
        selected_bucket = amount_buckets[0]
        for bucket in amount_buckets:
            cumsum += bucket.get("p", 0.0)
            if r <= cumsum:
                selected_bucket = bucket
                break

        # 从 bucket 的 mult_range 中采样倍数
        mult_range = selected_bucket.get("mult_range", [1.0, 1.0])
        if len(mult_range) >= 2:
            mult_min, mult_max = mult_range[0], mult_range[1]
            multiplier = self._rng.uniform(mult_min, mult_max)
        else:
            multiplier = 1.0

        # 使用倍数采样金额
        return self._sample_amounts(from_acc, to_acc, multiplier=multiplier)

    def _sample_payment_format(self) -> str:
        """
        采样支付方式

        Returns:
            支付方式字符串
        """
        # 优先调用 manager 的 sample_payment_format
        if self.manager_agent is not None:
            return self.manager_agent.sample_payment_format()
        elif self.manager is not None:
            return self.manager.sample_payment_format()
        else:
            # 备用实现：从固定集合采样
            return self._rng.choice(self.payment_formats)

    def _sample_payment_format_with_plan(self, payment_format_p: Dict[str, float]) -> str:
        """
        使用 payment_format_p 采样支付方式

        Args:
            payment_format_p: 支付方式概率分布

        Returns:
            支付方式字符串
        """
        # 如果没有提供概率分布，使用默认方法
        if not payment_format_p:
            return self._sample_payment_format()

        # 按概率加权采样
        formats = list(payment_format_p.keys())
        weights = list(payment_format_p.values())

        # 使用 CDF 采样
        total_weight = sum(weights)
        if total_weight <= 1e-6:
            return self._sample_payment_format()

        r = self._rng.uniform(0, total_weight)
        cumsum = 0.0
        for fmt, weight in zip(formats, weights):
            cumsum += weight
            if r <= cumsum:
                return fmt

        # 降级：返回最后一个
        return formats[-1] if formats else self._sample_payment_format()

    def _sample_currencies(self) -> tuple:
        """
        智能采样货币对（支持跨币种交易）

        策略：
        - 大部分交易使用默认货币（同币种）
        - 小部分交易使用跨币种（根据 cross_currency_ratio）
        - 跨币种交易时，从货币列表中随机选择不同的货币

        Returns:
            (payment_currency, receiving_currency) 元组
        """
        # 决定是否为跨币种交易
        is_cross_currency = self._rng.random() < self.cross_currency_ratio

        if not is_cross_currency:
            # 同币种交易：使用默认货币
            return (self.default_currency, self.default_currency)
        else:
            # 跨币种交易：从货币列表中选择两种不同的货币
            if len(self.payment_currencies) < 2:
                # 如果货币列表不足2种，降级为同币种
                return (self.default_currency, self.default_currency)

            # 采样支付货币
            payment_currency = self._rng.choice(self.payment_currencies)

            # 采样接收货币（确保与支付货币不同）
            receiving_candidates = [c for c in self.receiving_currencies if c != payment_currency]
            if receiving_candidates:
                receiving_currency = self._rng.choice(receiving_candidates)
            else:
                # 如果没有不同的货币可选，降级为同币种
                receiving_currency = payment_currency

            return (payment_currency, receiving_currency)

    def _calculate_amounts_with_exchange(
        self,
        base_amount: float,
        payment_currency: str,
        receiving_currency: str,
        from_bank: Optional[str] = None,
        to_bank: Optional[str] = None
    ) -> tuple:
        """
        根据汇率和手续费计算最终的支付金额和接收金额

        Args:
            base_amount: 基础金额（以默认货币计价）
            payment_currency: 支付货币
            receiving_currency: 接收货币
            from_bank: 发起方银行
            to_bank: 接收方银行

        Returns:
            (amount_paid, amount_received) 元组
        """
        # 判断是否为跨币种交易
        is_cross_currency = (payment_currency != receiving_currency)

        # 判断是否为同银行转账
        is_same_bank = (from_bank is not None and to_bank is not None and from_bank == to_bank)

        if not is_cross_currency:
            # 同币种交易：根据是否同银行决定手续费率
            amount_paid = base_amount
            if is_same_bank:
                # 同银行内转账：使用同银行手续费率（通常为0）
                fee = amount_paid * self.same_bank_fee_rate
            else:
                # 跨银行转账：使用同币种手续费率
                fee = amount_paid * self.same_currency_fee_rate
            amount_received = amount_paid - fee
            return (amount_paid, amount_received)

        # 跨币种交易：需要汇率转换和更高的手续费
        # 步骤 1: 将基础金额转换为支付货币
        # 正确的转换：base_amount (default_currency) -> USD -> payment_currency
        default_currency_rate = self.exchange_rates.get(self.default_currency, 1.0)
        payment_rate = self.exchange_rates.get(payment_currency, 1.0)
        amount_paid = base_amount * (payment_rate / default_currency_rate)

        # 步骤 2: 计算汇率（考虑价差）
        receiving_rate = self.exchange_rates.get(receiving_currency, 1.0)
        # 实际汇率 = 市场汇率 × (1 - 价差)
        effective_exchange_rate = (receiving_rate / payment_rate) * (1 - self.exchange_rate_spread)

        # 步骤 3: 转换为接收货币（扣除汇率价差）
        amount_after_exchange = amount_paid * effective_exchange_rate

        # 步骤 4: 扣除跨币种交易手续费
        fee = amount_after_exchange * self.cross_currency_fee_rate
        amount_received = amount_after_exchange - fee

        return (amount_paid, amount_received)

