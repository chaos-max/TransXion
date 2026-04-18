"""
TransactionManager: 交易数据管理器

职责：
- 数据初始化/恢复（load_data）
- 发起方采样（sample_senders）
- 对手选择（choose_counterparty）
- 交易写入与缓存更新（add_transactions）
- 保存与日志（save_infos）
- 续跑判断（rerun）
"""

import random
import os
import json
import csv
import bisect
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from LLMGraph.manager.base import BaseManager
from LLMGraph.manager import manager_registry
from LLMGraph.utils.data_generator import generate_account_id

@manager_registry.register("transaction")
class TransactionManager(BaseManager):
    """交易管理器（对应 SocialManager）"""

    def __init__(
        self,
        output_dir: str,
        num_accounts: int = 10_000,
        merchant_ratio: float = 0.2,
        num_merchants: Optional[int] = None,  # 新增:商户数量
        num_persons: Optional[int] = None,    # 新增:个人数量
        use_fixed_daily_generation: bool = False,  # 新增:是否使用固定每日生成模式
        p2m_ratio_for_person: float = 0.75,
        m2m_ratio_for_merchant: float = 0.7,
        exploration_alpha: float = 5.0,
        favorites_topk: int = 10,
        favorites_decay: float = 0.97,
        buffer_size: int = 10000,
        profiles_path: Optional[str] = None,
        merchants_path: Optional[str] = None,
        num_geos: int = 20,
        p_cross_base: float = 0.03,
        p_repeat_local: float = 0.6,
        p_repeat_external: float = 0.7,
        model_config_name: str = "profile_generator",  # 数据生成器使用的模型配置
        banks: Optional[List[str]] = None,  # 银行列表
        **kwargs
    ):
        super().__init__(**kwargs)

        # 基础配置
        self.output_dir = output_dir

        # 支持两种配置方式:
        # 1. 新方式:直接指定num_merchants和num_persons (可以是每天生成的固定数量)
        # 2. 旧方式:使用num_accounts和merchant_ratio计算 (累积总数)
        self.use_fixed_daily_generation = use_fixed_daily_generation  # 直接使用配置参数
        if num_merchants is not None and num_persons is not None:
            self.num_merchants = num_merchants
            self.num_persons = num_persons
            self.num_accounts = num_merchants + num_persons
            self.merchant_ratio = num_merchants / self.num_accounts if self.num_accounts > 0 else 0.2
        else:
            # 向后兼容旧配置
            self.num_accounts = num_accounts
            self.merchant_ratio = merchant_ratio
            self.num_merchants = int(num_accounts * merchant_ratio)
            self.num_persons = num_accounts - self.num_merchants

        self.model_config_name = model_config_name
        self.p2m_ratio_for_person = p2m_ratio_for_person
        self.m2m_ratio_for_merchant = m2m_ratio_for_merchant
        self.exploration_alpha = exploration_alpha
        self.favorites_topk = favorites_topk
        self.favorites_decay = favorites_decay
        self.buffer_size = buffer_size
        self.num_geos = num_geos
        self.p_cross_base = p_cross_base
        self.p_repeat_local = p_repeat_local
        self.p_repeat_external = p_repeat_external

        # 数据文件路径（可覆盖）
        self.profiles_path = profiles_path
        self.merchants_path = merchants_path

        # 默认数据路径（若未显式传入）
        if self.profiles_path is None or self.merchants_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # LLMGraph
            default_profiles = os.path.join(base, "tasks", "transaction", "data", "profiles.json")
            default_merchants = os.path.join(base, "tasks", "transaction", "data", "merchants.json")
            if self.profiles_path is None:
                self.profiles_path = default_profiles
            if self.merchants_path is None:
                self.merchants_path = default_merchants

        # 账户数据
        self.accounts = []
        self.account_map = {}  # O(1) 查找
        self.person_ids = []
        self.merchant_ids = []

        # 银行和支付方式（中国大陆）
        self.banks = banks if banks else ["ICBC", "CCB", "ABC", "BOC", "CMB"]
        # 中国大陆常用支付方式：
        # - Mobile: 移动支付（微信/支付宝），主要用于日常消费、扫码支付
        # - Card: 银行卡支付（借记卡/信用卡），用于刷卡消费
        # - Transfer: 银行转账，用于大额转账、工资发放等
        # - Cash: 现金支付
        self.payment_formats = ["Mobile", "Card", "Transfer", "Cash"]

        # geo 子图结构
        self.geo_persons = {}  # geo_id -> [person_id]
        self.geo_merchants = {}  # geo_id -> [merchant_id]
        self.geo_merchant_popularity = {}  # geo_id -> {merchant_id: weight}

        # 对手网络（local/external favorites）
        self.local_favorites = {}  # account_id -> [(counterparty_id, weight)]
        self.external_favorites = {}  # account_id -> [(counterparty_id, weight)]

        # 采样加速：CDF for sender sampling
        self.sender_weights = []
        self.sender_weights_cdf = []
        self.sender_ids = []

        # 商户权重（用于 hub 识别）
        self.merchant_popularity = {}
        self.top_merchants = []

        # 交易缓冲区
        self.transaction_buffer = []

        # 统计信息
        self.total_txn_generated = 0
        self.cur_time = None
        self.last_saved_time = None

        # 统计计数器
        self.stats = {
            "daily_txn_count": {},
            "p2m_count": 0,
            "p2p_count": 0,
            "m2m_count": 0,
            "m2p_count": 0,
            "cross_bank_count": 0,
            "format_distribution": {},
        }

        # 追踪已保存到CSV的账户ID（用于增量保存）
        self.saved_person_ids = set()
        self.saved_merchant_ids = set()

        # 性能统计
        self.simulation_start_time = time.time()
        self.round_times = []

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    async def load_data_async(cls, start_time: datetime, end_time: datetime, **kwargs):
        """
        初始化或恢复数据（对应 SocialManager.load_data）- 异步版本

        返回一个初始化好的 manager 实例
        """
        # 创建 manager 实例
        manager = cls(**kwargs)

        # 初始化时间（如果续跑会在 _load_state 中覆盖）
        manager.cur_time = start_time
        manager.last_saved_time = start_time

        # 检查是否续跑
        if manager.rerun():
            print("Resuming from previous state...")
            # 加载状态（但不加载账户）
            state_file = os.path.join(manager.output_dir, "state.json")
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 恢复基础状态
            manager.cur_time = datetime.strptime(state["cur_time"], "%Y-%m-%d %H:%M:%S")
            manager.total_txn_generated = state["total_txn_generated"]

            # 检查是否是旧格式（包含 accounts）
            if "accounts" in state:
                print("检测到旧格式 state.json，使用旧数据恢复...")
                manager.accounts = state["accounts"]
                if "geo_persons" in state:
                    manager.geo_persons = state["geo_persons"]
                if "geo_merchants" in state:
                    manager.geo_merchants = state["geo_merchants"]
            else:
                print("检测到新格式 state.json，从原始文件重新加载 accounts...")
                # 从 state 获取路径
                if "profiles_path" in state:
                    manager.profiles_path = state["profiles_path"]
                if "merchants_path" in state:
                    manager.merchants_path = state["merchants_path"]
                if "num_geos" in state:
                    manager.num_geos = state["num_geos"]

                # 使用异步方法重新加载账户
                await manager._generate_accounts_async(max_parallel=kwargs.get('max_parallel', 10))

            # 恢复其他状态
            if "geo_merchant_popularity" in state:
                manager.geo_merchant_popularity = state["geo_merchant_popularity"]

            manager.local_favorites = {}
            if "local_favorites" in state:
                for acc_id, fav_list in state["local_favorites"].items():
                    manager.local_favorites[acc_id] = [(item[0], item[1]) for item in fav_list]

            manager.external_favorites = {}
            if "external_favorites" in state:
                for acc_id, fav_list in state["external_favorites"].items():
                    manager.external_favorites[acc_id] = [(item[0], item[1]) for item in fav_list]

            if "merchant_popularity" in state:
                manager.merchant_popularity = state["merchant_popularity"]

            # 恢复CSV增量保存追踪集合（从state恢复）
            if "saved_person_ids" in state:
                manager.saved_person_ids = set(state["saved_person_ids"])
            if "saved_merchant_ids" in state:
                manager.saved_merchant_ids = set(state["saved_merchant_ids"])

            # 重要：从CSV同步已保存的ID，以CSV为准（避免state.json丢失导致重复保存）
            manager._sync_saved_ids_from_csv()

            # 为新加载的账户初始化 favorites（如果不存在）
            for acc_id in manager.account_map.keys():
                if acc_id not in manager.local_favorites:
                    manager.local_favorites[acc_id] = []
                if acc_id not in manager.external_favorites:
                    manager.external_favorites[acc_id] = []

            print(f"Resumed from {manager.cur_time}, total transactions: {manager.total_txn_generated}")
        else:
            print("Starting new simulation...")
            # 使用异步方法生成账户
            await manager._generate_accounts_async(max_parallel=kwargs.get('max_parallel', 10))
            manager._initialize_favorites()

        # 构建采样加速结构
        manager._build_sender_cdf()

        print(f"Loaded {len(manager.accounts)} accounts ({len(manager.person_ids)} persons, {len(manager.merchant_ids)} merchants)")

        return manager

    def rerun(self) -> bool:
        """判断是否续跑（对应 SocialManager.rerun）"""
        state_file = os.path.join(self.output_dir, "state.json")
        return os.path.exists(state_file) and os.path.getsize(state_file) > 0

    def get_start_time(self) -> str:
        """返回开始时间字符串"""
        if self.cur_time:
            return self.cur_time.strftime("%Y-%m-%d %H:%M:%S")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _generate_accounts_async(self, max_parallel: int = 10):
        """异步生成账户（个人+商户）- 并行版本

        优先从 transaction_output 的备份文件加载，如果不存在则从原始文件加载

        Args:
            max_parallel: 最大并行数（控制同时生成的账户数量）
        """
        from LLMGraph.utils.data_generator import TransactionDataGenerator

        # 优先使用 transaction_output 中的备份文件
        profiles_backup = os.path.join(self.output_dir, "profiles_backup.json")
        merchants_backup = os.path.join(self.output_dir, "merchants_backup.json")

        # 确定使用哪个文件
        if os.path.exists(profiles_backup):
            profiles_file = profiles_backup
            print(f"使用备份文件: {profiles_backup}")
        else:
            profiles_file = self.profiles_path
            print(f"使用原始文件: {self.profiles_path}")

        if os.path.exists(merchants_backup):
            merchants_file = merchants_backup
            print(f"使用备份文件: {merchants_backup}")
        else:
            merchants_file = self.merchants_path
            print(f"使用原始文件: {self.merchants_path}")

        # 验证和创建文件
        profiles_exists = os.path.exists(profiles_file)
        merchants_exists = os.path.exists(merchants_file)

        if not profiles_exists:
            print(f"创建新的profiles文件: {profiles_file}")
            os.makedirs(os.path.dirname(profiles_file), exist_ok=True)
            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump({"records": []}, f, ensure_ascii=False, indent=2)

        if not merchants_exists:
            print(f"创建新的merchants文件: {merchants_file}")
            os.makedirs(os.path.dirname(merchants_file), exist_ok=True)
            with open(merchants_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        # 读取现有数据
        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                profiles_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {profiles_file}: {e}")

        if isinstance(profiles_data, dict) and "records" in profiles_data:
            profiles = profiles_data["records"]
        elif isinstance(profiles_data, list):
            profiles = profiles_data
        else:
            profiles = []

        try:
            with open(merchants_file, "r", encoding="utf-8") as f:
                merchants_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {merchants_file}: {e}")

        if isinstance(merchants_data, dict) and "records" in merchants_data:
            merchants = merchants_data["records"]
        elif isinstance(merchants_data, list):
            merchants = merchants_data
        else:
            merchants = []

        print(f"现有数据: {len(profiles)} 个人, {len(merchants)} 个商户")

        # 根据模式决定生成策略
        skip_generation = False
        if self.use_fixed_daily_generation:
            # 固定每日生成模式：
            # - 如果从备份文件加载（恢复状态），则跳过生成
            # - 如果从原始文件加载（初始运行），则保持原有数据，不额外生成
            # - 每天的新增通过 add_new_accounts_async 完成
            if os.path.exists(profiles_backup) or os.path.exists(merchants_backup):
                print(f"检测到备份文件，跳过生成（固定每日生成模式）")
                need_generate_persons = 0
                need_generate_merchants = 0
                skip_generation = True
            else:
                # 从原始文件加载，保持原有数据，不额外生成
                print(f"从原始文件加载初始数据（固定每日生成模式）")
                need_generate_persons = 0
                need_generate_merchants = 0
                skip_generation = True
        else:
            # 累积总数模式：补充到目标总数
            print(f"配置要求: {self.num_persons} 个人, {self.num_merchants} 个商户")
            need_generate_persons = max(0, self.num_persons - len(profiles))
            need_generate_merchants = max(0, self.num_merchants - len(merchants))

        if need_generate_persons or need_generate_merchants:
            print("\n检测到数据不足,启动LLM生成器 (异步模式)...")

            # 异步生成个人数据
            if need_generate_persons:
                if self.use_fixed_daily_generation:
                    num_needed = need_generate_persons
                else:
                    num_needed = self.num_persons - len(profiles)
                print(f"\n需要生成 {num_needed} 个个人数据 (异步并行)...")

                # 为个人数据创建专门的生成器
                person_generator = TransactionDataGenerator(model_config_name="profile_generator")
                new_persons = await self._generate_persons_parallel(
                    person_generator, num_needed, len(profiles), max_parallel
                )
                profiles.extend(new_persons)

                # 保存到备份文件
                if isinstance(profiles_data, dict) and "records" in profiles_data:
                    profiles_data["records"] = profiles
                    save_data = profiles_data
                else:
                    save_data = {"records": profiles}

                with open(profiles_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存 {len(profiles)} 个个人数据到 {profiles_file}")

            # 异步生成商户数据
            if need_generate_merchants:
                if self.use_fixed_daily_generation:
                    num_needed = need_generate_merchants
                else:
                    num_needed = self.num_merchants - len(merchants)
                print(f"\n需要生成 {num_needed} 个商户数据 (异步并行)...")

                existing_person_ids = []
                for item in profiles:
                    if isinstance(item, dict) and "person" in item:
                        person_id = item["person"].get("person_id")
                        if person_id:
                            existing_person_ids.append(person_id)

                # 为商户数据创建专门的生成器
                merchant_generator = TransactionDataGenerator(model_config_name="merchant_generator")
                new_merchants = await self._generate_merchants_parallel(
                    merchant_generator, num_needed, len(merchants), existing_person_ids, max_parallel
                )
                merchants.extend(new_merchants)

                # 保存到备份文件
                if isinstance(merchants_data, dict) and "records" in merchants_data:
                    merchants_data["records"] = merchants
                    save_data = merchants_data
                else:
                    save_data = merchants

                with open(merchants_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存 {len(merchants)} 个商户数据到 {merchants_file}")

        # 后续处理
        # 注意：固定每日生成模式下不截断，保留所有累积数据
        if not self.use_fixed_daily_generation:
            profiles = profiles[:self.num_persons]
            merchants = merchants[:self.num_merchants]
        else:
            # 固定每日生成模式：如果从原始文件加载，需要创建备份文件
            if not os.path.exists(profiles_backup) and profiles:
                print(f"创建个人数据备份文件: {profiles_backup}")
                if isinstance(profiles_data, dict) and "records" in profiles_data:
                    save_data = profiles_data
                else:
                    save_data = {"records": profiles}
                with open(profiles_backup, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存 {len(profiles)} 个个人数据到备份文件")

            if not os.path.exists(merchants_backup) and merchants:
                print(f"创建商户数据备份文件: {merchants_backup}")
                if isinstance(merchants_data, dict) and "records" in merchants_data:
                    save_data = merchants_data
                else:
                    save_data = merchants
                with open(merchants_backup, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存 {len(merchants)} 个商户数据到备份文件")

        # 如果CSV文件已存在，从CSV读取已保存的ID（避免重复保存）
        # 必须在处理账户之前读取，这样 saved_*_ids 才能正确初始化
        self._sync_saved_ids_from_csv()

        self.accounts = []

        # 处理账户数据（复用同步版本的逻辑）
        self._process_accounts_data(profiles, merchants)

    async def _generate_persons_parallel(
        self, generator, num_needed: int, start_id: int, max_parallel: int, batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        并行生成个人数据（支持批量生成，带重试机制）

        Args:
            generator: 数据生成器实例
            num_needed: 需要生成的总人数
            start_id: 起始ID
            max_parallel: 最大并行任务数
            batch_size: 每个API调用生成的人数（默认10）
        """
        async def generate_batch_persons_with_retry(batch_idx: int, batch_count: int, max_retries: int = 3):
            """生成一批个人数据（带重试）"""
            loop = asyncio.get_event_loop()
            for attempt in range(max_retries):
                try:
                    result = await loop.run_in_executor(
                        None,
                        generator.generate_persons,
                        batch_count,  # 每次生成batch_count个
                        start_id + batch_idx * batch_size,
                        batch_size  # 传递batch_size参数
                    )
                    if result:
                        return result
                    else:
                        print(f"  警告: 批次 {batch_idx} 生成失败 (尝试 {attempt + 1}/{max_retries})")
                except Exception as e:
                    print(f"  错误: 批次 {batch_idx} 生成异常: {e} (尝试 {attempt + 1}/{max_retries})")

                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

            print(f"  失败: 批次 {batch_idx} 在 {max_retries} 次尝试后仍然失败")
            return None

        all_persons = []

        # 计算需要多少批次
        total_batches = (num_needed + batch_size - 1) // batch_size

        # 按max_parallel分组执行批次
        for group_start in range(0, total_batches, max_parallel):
            group_end = min(group_start + max_parallel, total_batches)

            # 创建本组的任务
            tasks = []
            for batch_idx in range(group_start, group_end):
                # 计算本批次实际需要生成的人数
                remaining = num_needed - batch_idx * batch_size
                current_batch_size = min(batch_size, remaining)

                if current_batch_size > 0:
                    tasks.append(generate_batch_persons_with_retry(batch_idx, current_batch_size))

            # 并行执行本组任务
            batch_results = await asyncio.gather(*tasks)

            # 收集结果
            for result in batch_results:
                if result:
                    all_persons.extend(result)

            # 输出进度
            progress = len(all_persons) / num_needed * 100
            print(f"  生成个人数据进度: {len(all_persons)}/{num_needed} ({progress:.1f}%)")

        # 如果生成数量不足，补充生成
        if len(all_persons) < num_needed:
            shortage = num_needed - len(all_persons)
            print(f"  检测到生成不足，需要补充 {shortage} 个账户...")
            next_id = start_id + total_batches * batch_size
            补充_persons = await self._generate_persons_parallel(
                generator, shortage, next_id, max_parallel, batch_size
            )
            all_persons.extend(补充_persons)

        return all_persons

    async def _generate_merchants_parallel(
        self, generator, num_needed: int, start_id: int,
        existing_person_ids: List[str], max_parallel: int
    ) -> List[Dict[str, Any]]:
        """并行生成商户数据（带重试机制）"""
        async def generate_single_merchant_with_retry(idx: int, max_retries: int = 3):
            """生成单个商户数据（带重试）"""
            loop = asyncio.get_event_loop()
            for attempt in range(max_retries):
                try:
                    result = await loop.run_in_executor(
                        None,
                        generator.generate_merchants,
                        1,  # 每次生成1个
                        start_id + idx,
                        existing_person_ids
                    )
                    if result:
                        return result
                    else:
                        print(f"  警告: 商户 {idx} 生成失败 (尝试 {attempt + 1}/{max_retries})")
                except Exception as e:
                    print(f"  错误: 商户 {idx} 生成异常: {e} (尝试 {attempt + 1}/{max_retries})")

                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

            print(f"  失败: 商户 {idx} 在 {max_retries} 次尝试后仍然失败")
            return None

        all_merchants = []
        for batch_start in range(0, num_needed, max_parallel):
            batch_size = min(max_parallel, num_needed - batch_start)
            tasks = [generate_single_merchant_with_retry(batch_start + i) for i in range(batch_size)]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if result:
                    all_merchants.extend(result)

            # 每20个输出一次进度
            if len(all_merchants) % 20 == 0 or len(all_merchants) == num_needed:
                progress = len(all_merchants) / num_needed * 100
                print(f"  生成商户数据进度: {len(all_merchants)}/{num_needed} ({progress:.1f}%)")

        # 如果生成数量不足，补充生成
        if len(all_merchants) < num_needed:
            shortage = num_needed - len(all_merchants)
            print(f"  检测到商户生成不足，需要补充 {shortage} 个...")
            next_id = start_id + num_needed
            additional_merchants = await self._generate_merchants_parallel(
                generator, shortage, next_id, existing_person_ids, max_parallel
            )
            all_merchants.extend(additional_merchants)

        return all_merchants

    def _sync_saved_ids_from_csv(self):
        """
        从CSV文件同步已保存的账户ID

        这个方法确保 saved_person_ids 和 saved_merchant_ids 与CSV文件保持一致，
        避免因 state.json 丢失或损坏导致重复保存账户。
        """
        import csv

        persons_csv = os.path.join(self.output_dir, "persons.csv")
        merchants_csv = os.path.join(self.output_dir, "merchants.csv")

        if os.path.exists(persons_csv):
            with open(persons_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    person_id = row.get("person_id")
                    if person_id:
                        self.saved_person_ids.add(person_id)
            print(f"✓ 从 persons.csv 同步了 {len(self.saved_person_ids)} 个已保存的个人ID")

        if os.path.exists(merchants_csv):
            with open(merchants_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    merchant_id = row.get("merchant_id")
                    if merchant_id:
                        self.saved_merchant_ids.add(merchant_id)
            print(f"✓ 从 merchants.csv 同步了 {len(self.saved_merchant_ids)} 个已保存的商户ID")

    def _generate_bank_account_number(self, account_id: str, idx: int) -> str:
        """
        生成银行账户编号（以a开头）

        Args:
            account_id: 账户ID（person_id或merchant_id）
            idx: 账户索引

        Returns:
            银行账户编号，格式：a + 8位hash（如 a169B5B82，总共9位）
        """
        return generate_account_id("A", idx)

    def _process_accounts_data(self, profiles: List[Dict], merchants: List[Dict]):
        """处理账户数据的公共逻辑（同步和异步版本共用）"""
        # 处理个人账户
        for idx, item in enumerate(profiles):
            if not isinstance(item, dict):
                print(f"警告: 跳过无效的个人数据 (索引 {idx})")
                continue

            if "person" not in item or "tags" not in item:
                print(f"警告: 跳过缺少字段的个人数据 (索引 {idx})")
                continue

            person = item["person"]
            tags = item["tags"]

            if not isinstance(person, dict):
                print(f"警告: 跳过无效的person字段 (索引 {idx})")
                continue

            person_id = person.get("person_id", generate_account_id("P", idx))
            account = {
                "account_id": person_id,
                "bank_account_number": self._generate_bank_account_number(person_id, idx),
                "account_type": "person",
                "name": person.get("name", f"Person_{idx}"),
                "geo_id": person.get("geo_id", idx % self.num_geos),
                "balance": person.get("balance", 10000.0),
                "bank": random.choice(self.banks),  # 随机分配银行
                "person": person,  # 保存原始person对象，用于导出账户映射表
                "tags": tags,
                "txn_count_as_from": 0,  # 作为发起方的交易计数
                "txn_count_as_to": 0     # 作为接收方的交易计数
            }
            self.accounts.append(account)

        # 处理商户账户
        person_count = len(profiles)  # 获取个人账户数量，用于计算商户账户编号偏移
        for idx, item in enumerate(merchants):
            if not isinstance(item, dict):
                print(f"警告: 跳过无效的商户数据 (索引 {idx})")
                continue

            # 支持两种数据结构：
            # 1. 原始数据: {"company": {"Company_id": "c000001", ...}, "tags": {...}}
            # 2. 新生成数据: {"merchant_id": "...", "merchant_name": "...", ...}
            if "company" in item:
                # 原始数据结构
                company = item.get("company", {})
                merchant_id = company.get("Company_id", generate_account_id("C", idx))
                merchant_name = company.get("Company_description", f"Merchant_{idx}")
                geo_id = item.get("geo_id", idx % self.num_geos)
                balance = company.get("balance", 50000.0)
                tags = item.get("tags", {})
                merchant_obj = item  # 保存完整的原始对象
            else:
                # 新生成的数据结构
                merchant_id = item.get("merchant_id", generate_account_id("C", idx))
                merchant_name = item.get("merchant_name", f"Merchant_{idx}")
                geo_id = item.get("geo_id", idx % self.num_geos)
                balance = item.get("balance", 50000.0)
                tags = item.get("tags", {})
                merchant_obj = item

            account_idx = person_count + idx  # 商户账户编号从个人账户数量之后开始
            account = {
                "account_id": merchant_id,
                "bank_account_number": self._generate_bank_account_number(merchant_id, account_idx),
                "account_type": "merchant",
                "name": merchant_name,
                "geo_id": geo_id,
                "balance": balance,
                "bank": random.choice(self.banks),  # 随机分配银行
                "merchant": merchant_obj,  # 保存原始merchant对象，用于导出账户映射表
                "tags": tags,
                "txn_count_as_from": 0,  # 作为发起方的交易计数
                "txn_count_as_to": 0     # 作为接收方的交易计数
            }
            self.accounts.append(account)

        print(f"✓ 总共加载 {len(self.accounts)} 个账户")

        # 将 bank_account_number 和 bank 信息写回到备份文件
        self._update_backup_files_with_bank_info(profiles, merchants)

        # 初始化 geo 子图
        for geo_id in range(self.num_geos):
            self.geo_persons[geo_id] = []
            self.geo_merchants[geo_id] = []
            self.geo_merchant_popularity[geo_id] = {}

        for account in self.accounts:
            acc_id = account["account_id"]
            self.account_map[acc_id] = account
            geo_id = account.get("geo_id", 0)

            if account["account_type"] == "person":
                self.person_ids.append(acc_id)
                self.geo_persons[geo_id].append(acc_id)
            elif account["account_type"] == "merchant":
                self.merchant_ids.append(acc_id)
                self.geo_merchants[geo_id].append(acc_id)
                self.geo_merchant_popularity[geo_id][acc_id] = 0.0

    def _initialize_favorites(self):
        """初始化 favorites 容器"""
        self.local_favorites = {}
        self.external_favorites = {}

    def _build_sender_cdf(self):
        """构建发起方采样的 CDF（累计分布函数）用于 O(log N) 采样"""
        self.sender_ids = []
        self.sender_weights = []

        for account in self.accounts:
            self.sender_ids.append(account["account_id"])
            self.sender_weights.append(account["tags"]["avg_txn_cnt_daily"])

        # 构建 CDF
        self.sender_weights_cdf = []
        cumsum = 0.0
        for w in self.sender_weights:
            cumsum += w
            self.sender_weights_cdf.append(cumsum)

    # ==================== 采样方法 ====================

    def sample_senders(self, txn_count: int) -> List[str]:
        """
        采样发起方账户（对应 SocialManager.sample_cur_agents_llmplan）

        使用 CDF + 二分查找，复杂度 O(txn_count * log N)
        """
        if txn_count == 0 or not self.sender_weights_cdf:
            return []

        total_weight = self.sender_weights_cdf[-1]
        sampled_ids = []

        for _ in range(txn_count):
            r = random.uniform(0, total_weight)
            idx = bisect.bisect_left(self.sender_weights_cdf, r)
            if idx >= len(self.sender_ids):
                idx = len(self.sender_ids) - 1
            sampled_ids.append(self.sender_ids[idx])

        return sampled_ids

    def choose_counterparty(self, from_account: Dict[str, Any], cross: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        选择交易对手

        Args:
            from_account: 发起方账户
            cross: 是否跨 geo（None=自动决定，True=跨geo，False=本地）

        Returns:
            对手账户，或 None
        """
        from_id = from_account["account_id"]
        from_type = from_account["account_type"]
        from_geo = from_account.get("geo_id", 0)

        # 决定对手类型（P2M/P2P 或 M2M/M2P）
        if from_type == "person":
            to_merchant = random.random() < self.p2m_ratio_for_person
        else:
            to_merchant = random.random() < self.m2m_ratio_for_merchant

        # 决定是否跨 geo
        if cross is None:
            # 自动决定：基于 p_cross_base
            cross = random.random() < self.p_cross_base

        if cross:
            # 跨 geo 采样
            return self._choose_external_counterparty(from_id, from_geo, to_merchant)
        else:
            # 本地 geo 采样
            return self._choose_local_counterparty(from_id, from_geo, to_merchant)

    def _choose_local_counterparty(self, from_id: str, from_geo: int, to_merchant: bool) -> Optional[Dict[str, Any]]:
        """本地 geo 内选择对手"""
        local_fav = self.local_favorites.get(from_id, [])

        # 过滤符合类型的 local favorites
        valid_fav = []
        for counterparty_id, weight in local_fav:
            counterparty = self.account_map.get(counterparty_id)
            if counterparty is None:
                continue
            if (counterparty["account_type"] == "merchant") == to_merchant:
                valid_fav.append((counterparty, weight))

        # 以 p_repeat_local 概率从 favorites 采样
        if valid_fav and random.random() < self.p_repeat_local:
            return self._weighted_sample(valid_fav)

        # 否则从 geo 内商户/个人池采样（按热度）
        return self._sample_from_geo_pool(from_id, from_geo, to_merchant)

    def _choose_external_counterparty(self, from_id: str, from_geo: int, to_merchant: bool) -> Optional[Dict[str, Any]]:
        """跨 geo 选择对手"""
        external_fav = self.external_favorites.get(from_id, [])

        # 过滤符合类型的 external favorites
        valid_fav = []
        for counterparty_id, weight in external_fav:
            counterparty = self.account_map.get(counterparty_id)
            if counterparty is None:
                continue
            if (counterparty["account_type"] == "merchant") == to_merchant:
                valid_fav.append((counterparty, weight))

        # 以 p_repeat_external 概率从 favorites 采样
        if valid_fav and random.random() < self.p_repeat_external:
            return self._weighted_sample(valid_fav)

        # 否则随机选择一个不同的 geo，然后从该 geo 内采样
        other_geos = [g for g in range(self.num_geos) if g != from_geo]
        if not other_geos:
            return self._sample_from_geo_pool(from_id, from_geo, to_merchant)

        target_geo = random.choice(other_geos)
        return self._sample_from_geo_pool(from_id, target_geo, to_merchant)

    def _weighted_sample(self, weighted_items: List[Tuple[Dict[str, Any], float]]) -> Optional[Dict[str, Any]]:
        """加权采样"""
        if not weighted_items:
            return None

        total_weight = sum(w for _, w in weighted_items)
        if total_weight <= 0:
            return weighted_items[0][0]

        r = random.uniform(0, total_weight)
        cumsum = 0
        for item, weight in weighted_items:
            cumsum += weight
            if r <= cumsum:
                return item
        return weighted_items[0][0]

    def _sample_from_geo_pool(self, from_id: str, geo_id: int, to_merchant: bool) -> Optional[Dict[str, Any]]:
        """从指定 geo 内按热度采样对手"""
        if to_merchant:
            pool = self.geo_merchants.get(geo_id, [])
            popularity = self.geo_merchant_popularity.get(geo_id, {})
        else:
            pool = self.geo_persons.get(geo_id, [])
            popularity = {}

        if not pool:
            # 降级到全局池
            return self._sample_from_global_pool(from_id, to_merchant)

        # 如果有热度数据，按热度加权采样
        if popularity and to_merchant:
            weights = [popularity.get(mid, 1.0) for mid in pool]
            total_weight = sum(weights)
            if total_weight > 0:
                r = random.uniform(0, total_weight)
                cumsum = 0
                for mid, w in zip(pool, weights):
                    if mid == from_id:
                        continue
                    cumsum += w
                    if r <= cumsum:
                        return self.account_map.get(mid)

        # 否则均匀采样（排除自己）
        for _ in range(10):
            to_id = random.choice(pool)
            if to_id != from_id:
                return self.account_map.get(to_id)

        # 如果10次都是自己，返回第一个不是自己的
        for to_id in pool:
            if to_id != from_id:
                return self.account_map.get(to_id)

        return None

    def _sample_from_global_pool(self, from_id: str, to_merchant: bool) -> Optional[Dict[str, Any]]:
        """从全局池采样对手（O(1) 索引）"""
        pool = self.merchant_ids if to_merchant else self.person_ids

        if not pool:
            pool = self.person_ids if to_merchant else self.merchant_ids

        if not pool:
            return None

        # 排除自己（最多尝试 10 次）
        for _ in range(10):
            to_id = random.choice(pool)
            if to_id != from_id:
                return self.account_map.get(to_id)

        # 如果 10 次都是自己，返回第一个不是自己的
        for to_id in pool:
            if to_id != from_id:
                return self.account_map.get(to_id)

        return None


    def sample_timestamp(self, window_start: datetime, window_end: datetime, tod_p: List[float]) -> datetime:
        """在窗口内采样秒级时间戳（基于 tod_p 分布）"""
        window_seconds = int((window_end - window_start).total_seconds())

        if window_seconds <= 0:
            return window_start

        # 如果窗口跨天，简化处理：在窗口内按 tod_p 采样
        if window_end.date() > window_start.date():
            offset_seconds = random.randint(0, window_seconds - 1)
            return window_start + timedelta(seconds=offset_seconds)

        # 计算窗口内每分钟的权重（基于 tod_p）
        minute_weights = []
        for minute in range(int(window_seconds / 60) + 1):
            current_time = window_start + timedelta(minutes=minute)
            if current_time >= window_end:
                break
            hour = current_time.hour
            segment = hour // 3  # 每3小时一个段
            weight = tod_p[segment] if segment < len(tod_p) else 0.0
            minute_weights.append(weight)

        # 如果所有权重都是0，使用均匀分布
        total_weight = sum(minute_weights)
        if total_weight <= 0:
            offset_seconds = random.randint(0, window_seconds - 1)
            return window_start + timedelta(seconds=offset_seconds)

        # 根据权重采样分钟
        r = random.random() * total_weight
        cumsum = 0.0
        selected_minute = 0
        for i, weight in enumerate(minute_weights):
            cumsum += weight
            if r <= cumsum:
                selected_minute = i
                break

        # 在选中的分钟内随机采样秒
        sampled_time = window_start + timedelta(minutes=selected_minute, seconds=random.randint(0, 59))

        # 确保不超过窗口结束时间
        if sampled_time >= window_end:
            sampled_time = window_end - timedelta(seconds=1)

        return sampled_time

    def _sample_time_segment(self, tod_p: List[float]) -> int:
        """根据 tod_p 采样时间段（0-7）"""
        r = random.random()
        cumsum = 0
        for i, p in enumerate(tod_p):
            cumsum += p
            if r <= cumsum:
                return i
        return len(tod_p) - 1

    def sample_amounts(self, from_account: Dict[str, Any], to_account: Dict[str, Any]) -> Tuple[float, float]:
        """采样交易金额（Amount Paid 和 Amount Received）- 优化版，支持商户类型"""
        # 获取基础平均金额
        avg_amt = from_account["tags"]["avg_txn_amt"]

        # 根据接收方商户类型调整金额范围
        merchant_type = to_account.get("type", None)

        if merchant_type == "Large":
            # Large商户：大额交易（1000-50000元）
            # 使用更大的基础金额和标准差
            base_amt = max(avg_amt, 5000.0)  # 至少5000元
            std = base_amt * 1.5  # 更大的标准差，增加大额交易概率
            # 10%概率生成超大额交易
            if random.random() < 0.1:
                base_amt *= random.uniform(3.0, 10.0)
        elif merchant_type == "Medium":
            # Medium商户：中等交易（100-5000元）
            base_amt = max(avg_amt, 500.0)  # 至少500元
            std = base_amt * 1.0
            # 5%概率生成较大额交易
            if random.random() < 0.05:
                base_amt *= random.uniform(2.0, 5.0)
        elif merchant_type == "Small":
            # Small商户：小额交易（10-500元）
            base_amt = min(avg_amt, 200.0)  # 最多200元
            std = base_amt * 0.8
        else:
            # P2P交易：支持常规交易和大额场景
            # 5%概率生成大额P2P交易（房租、借贷、二手交易、大额转账等）
            if random.random() < 0.05:
                # 大额P2P场景：5000-50000元
                base_amt = random.uniform(5000.0, 50000.0)
                std = base_amt * 0.3  # 较小的标准差，保持在合理范围
            else:
                # 常规P2P交易：基于发起方的avg_txn_amt
                base_amt = avg_amt
                std = avg_amt * 0.6  # 稍大的标准差，增加变化性

        # 正态分布采样
        amount_paid = abs(random.gauss(base_amt, std))
        amount_paid = max(1.0, amount_paid)

        fee_rate = random.uniform(0.0, 0.01)
        amount_received = amount_paid * (1 - fee_rate)

        return amount_paid, amount_received

    def sample_payment_format(self) -> str:
        """采样支付方式"""
        return random.choice(self.payment_formats)

    # ==================== 交易写入与统计 ====================

    def add_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """
        批量添加交易（对应 SocialManager.add_tweets）

        更新：buffer、favorites、统计计数
        """
        if not transactions:
            return 0

        for txn in transactions:
            self.transaction_buffer.append(txn)
            self.total_txn_generated += 1

            # 更新 favorites
            from_id = txn["From Account"]
            to_id = txn["To Account"]
            self._update_favorites(from_id, to_id)

            # 更新交易计数
            if from_id in self.account_map:
                self.account_map[from_id]["txn_count_as_from"] = \
                    self.account_map[from_id].get("txn_count_as_from", 0) + 1
            if to_id in self.account_map:
                self.account_map[to_id]["txn_count_as_to"] = \
                    self.account_map[to_id].get("txn_count_as_to", 0) + 1

            # 更新统计
            self._update_stats(txn)

        # 刷新缓冲区
        self._flush_buffer_if_needed()

        return len(transactions)

    def _update_favorites(self, from_id: str, to_id: str):
        """更新 favorites 列表（区分 local/external，应用衰减）"""
        from_acc = self.account_map.get(from_id)
        to_acc = self.account_map.get(to_id)
        if not from_acc or not to_acc:
            return

        from_geo = from_acc.get("geo_id", 0)
        to_geo = to_acc.get("geo_id", 0)
        is_cross = (from_geo != to_geo)

        # 选择目标集合
        if is_cross:
            favorites_dict = self.external_favorites
        else:
            favorites_dict = self.local_favorites

        favorites_list = favorites_dict.get(from_id, [])

        # 应用衰减到所有已有项
        favorites_list = [(cid, w * self.favorites_decay) for cid, w in favorites_list]

        # 查找是否已存在
        found = False
        for i, (counterparty_id, weight) in enumerate(favorites_list):
            if counterparty_id == to_id:
                favorites_list[i] = (counterparty_id, weight + 1.0)
                found = True
                break

        if not found:
            favorites_list.append((to_id, 1.0))

        # 按权重排序，保留 topK
        favorites_list.sort(key=lambda x: x[1], reverse=True)
        favorites_dict[from_id] = favorites_list[:self.favorites_topk]

    def _update_stats(self, txn: Dict[str, Any]):
        """更新统计信息"""
        # 按日期统计
        date_str = txn["Timestamp"][:10]
        if date_str not in self.stats["daily_txn_count"]:
            self.stats["daily_txn_count"][date_str] = 0
        self.stats["daily_txn_count"][date_str] += 1

        # 交易类型统计
        from_id = txn["From Account"]
        to_id = txn["To Account"]
        from_acc = self.account_map.get(from_id)
        to_acc = self.account_map.get(to_id)

        if from_acc and to_acc:
            from_type = from_acc["account_type"]
            to_type = to_acc["account_type"]

            if from_type == "person" and to_type == "merchant":
                self.stats["p2m_count"] += 1
            elif from_type == "person" and to_type == "person":
                self.stats["p2p_count"] += 1
            elif from_type == "merchant" and to_type == "merchant":
                self.stats["m2m_count"] += 1
            elif from_type == "merchant" and to_type == "person":
                self.stats["m2p_count"] += 1

        # 跨行统计
        if txn["From Bank"] != txn["To Bank"]:
            self.stats["cross_bank_count"] += 1

        # 支付方式统计
        payment_format = txn["Payment Format"]
        if payment_format not in self.stats["format_distribution"]:
            self.stats["format_distribution"][payment_format] = 0
        self.stats["format_distribution"][payment_format] += 1

        # 更新商户权重（收款笔数，全局 + geo）
        if to_acc and to_acc["account_type"] == "merchant":
            to_geo = to_acc.get("geo_id", 0)

            # 全局热度
            if to_id not in self.merchant_popularity:
                self.merchant_popularity[to_id] = 0.0
            self.merchant_popularity[to_id] += 1.0

            # geo 内热度
            if to_geo in self.geo_merchant_popularity:
                if to_id not in self.geo_merchant_popularity[to_geo]:
                    self.geo_merchant_popularity[to_geo][to_id] = 0.0
                self.geo_merchant_popularity[to_geo][to_id] += 1.0

    def update_top_merchants(self, top_k: int = 20):
        """更新 top merchants 列表（对应 SocialManager.update_big_name_list）"""
        if not self.merchant_popularity:
            return

        sorted_merchants = sorted(
            self.merchant_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.top_merchants = [m_id for m_id, _ in sorted_merchants[:top_k]]

    def _flush_buffer_if_needed(self, force: bool = False):
        """刷新缓冲区到文件（按日期分片）"""
        if len(self.transaction_buffer) < self.buffer_size and not force:
            return

        if len(self.transaction_buffer) == 0:
            return

        # 按日期分组
        txn_by_date = defaultdict(list)
        for txn in self.transaction_buffer:
            date_str = txn["Timestamp"][:10].replace("-", "").replace("/", "")  # YYYYMMDD
            txn_by_date[date_str].append(txn)

        # 分片写入
        for date_str, txns in txn_by_date.items():
            output_file = os.path.join(self.output_dir, f"transactions_{date_str}.csv")
            file_exists = os.path.exists(output_file)

            with open(output_file, "a", encoding="utf-8") as f:
                # 写入表头
                if not file_exists:
                    header = [
                        "Timestamp",
                        "From Bank",
                        "From Account",
                        "To Bank",
                        "To Account",
                        "Amount Received",
                        "Receiving Currency",
                        "Amount Paid",
                        "Payment Currency",
                        "Payment Format",
                    ]
                    f.write(",".join(header) + "\n")

                # 写入数据
                for txn in txns:
                    row = [
                        txn["Timestamp"],
                        txn["From Bank"],
                        txn["From Account"],
                        txn["To Bank"],
                        txn["To Account"],
                        str(txn["Amount Received"]),
                        txn["Receiving Currency"],
                        str(txn["Amount Paid"]),
                        txn["Payment Currency"],
                        txn["Payment Format"],
                    ]
                    f.write(",".join(row) + "\n")

        print(f"Flushed {len(self.transaction_buffer)} transactions to {len(txn_by_date)} files")

        # 清空缓冲区
        self.transaction_buffer = []

    # ==================== 保存与日志 ====================

    def save_infos(self, cur_time: datetime, start_time: datetime, force: bool = False):
        """
        保存状态和日志（对应 SocialManager.save_infos）

        输出文件：
        - state.json: 可恢复状态
        - ex_logs.json: 统计日志
        - transactions_YYYYMMDD.csv: 分片交易记录
        - persons.csv: 个人账户表
        - merchants.csv: 商户账户表
        - account_mapping.csv: 账户映射表
        """
        # 强制刷新缓冲区
        self._flush_buffer_if_needed(force=True)

        # 保存 state.json
        self._save_state(cur_time)

        # 保存 ex_logs.json
        self._save_ex_logs(cur_time, start_time)

        # 保存账户表
        self._save_account_tables()

        print(f"Saved state, logs, and account tables at {cur_time}")

    def _save_state(self, cur_time: datetime):
        """保存可恢复状态到 state.json

        只保存动态变化的状态，静态数据（accounts, geo结构）从原始文件重新加载
        """
        state = {
            # 基础状态
            "cur_time": cur_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_txn_generated": self.total_txn_generated,

            # 动态状态（必需）
            "local_favorites": {
                acc_id: [[item[0], item[1]] for item in fav_list]
                for acc_id, fav_list in self.local_favorites.items()
            },
            "external_favorites": {
                acc_id: [[item[0], item[1]] for item in fav_list]
                for acc_id, fav_list in self.external_favorites.items()
            },
            "stats": self.stats,
            "merchant_popularity": self.merchant_popularity,
            "geo_merchant_popularity": self.geo_merchant_popularity,

            # CSV增量保存追踪（用于恢复后继续增量模式）
            "saved_person_ids": list(self.saved_person_ids),
            "saved_merchant_ids": list(self.saved_merchant_ids),

            # 数据文件路径（用于续跑时重新加载）
            "profiles_path": self.profiles_path,
            "merchants_path": self.merchants_path,
            "num_geos": self.num_geos,
        }

        state_file = os.path.join(self.output_dir, "state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _save_ex_logs(self, cur_time: datetime, start_time: datetime):
        """保存统计日志到 ex_logs.json"""
        simulation_time = time.time() - self.simulation_start_time

        # 计算统计摘要
        total_txn = self.total_txn_generated
        stats_summary = {
            "total_transactions": total_txn,
            "p2m_ratio": self.stats["p2m_count"] / total_txn if total_txn > 0 else 0,
            "p2p_ratio": self.stats["p2p_count"] / total_txn if total_txn > 0 else 0,
            "m2m_ratio": self.stats["m2m_count"] / total_txn if total_txn > 0 else 0,
            "m2p_ratio": self.stats["m2p_count"] / total_txn if total_txn > 0 else 0,
            "cross_bank_ratio": self.stats["cross_bank_count"] / total_txn if total_txn > 0 else 0,
            "format_distribution": self.stats["format_distribution"],
            "top_merchants": self.top_merchants[:10],
        }

        ex_logs = {
            "simulation_time": simulation_time,
            "total_txn_generated": total_txn,
            "cur_time": cur_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "cur_account_num": len(self.accounts),
            "round_times": self.round_times,
            "stats_summary": stats_summary,
        }

        ex_logs_file = os.path.join(self.output_dir, "ex_logs.json")
        with open(ex_logs_file, "w", encoding="utf-8") as f:
            json.dump(ex_logs, f, indent=2, ensure_ascii=False)

    def _save_account_tables(self):
        """
        导出账户表为CSV格式（增量追加模式）

        第一次保存：写入所有账户（包括初始的100,000人+200商户）
        后续保存：只追加新增的账户

        输出文件:
        - persons.csv: 个人账户表
        - merchants.csv: 商户账户表
        - account_mapping.csv: 账户映射表
        """
        # 字段转换字典
        FIELD_DESCRIPTIONS = {
            "gender": {0: "female", 1: "male"},
            "occupation": {
                0: "student",
                1: "white-collar worker",
                2: "blue-collar worker",
                3: "freelancer",
                4: "business owner",
                5: "retired",
                6: "unemployed",
                7: "other"
            },
            "marital_status": {0: "single", 1: "married"},
            "education": {
                0: "below high school",
                1: "high school",
                2: "associate degree",
                3: "bachelor's degree",
                4: "graduate degree or higher"
            }
        }

        # 直接从 account_map 获取需要保存的账户（未保存过的）
        new_profiles = []
        new_merchants = []

        for account in self.accounts:
            acc_id = account.get("account_id")
            if account.get("account_type") == "person":
                if acc_id not in self.saved_person_ids:
                    new_profiles.append(account)
            elif account.get("account_type") == "merchant":
                if acc_id not in self.saved_merchant_ids:
                    new_merchants.append(account)

        # 1. 导出个人账户表（增量追加）
        persons_file = os.path.join(self.output_dir, "persons.csv")
        is_first_save = not os.path.exists(persons_file)

        if new_profiles:
            # 收集所有字段
            if is_first_save:
                # 第一次保存：收集所有字段
                person_fieldnames = set(["person_id", "bank_account_number", "bank", "is_deleted", "deleted_at"])
                for account in new_profiles:
                    person = account.get("person", {})
                    for k in person.keys():
                        if k != "person_id":
                            person_fieldnames.add(f"person_{k}")
                    tags = account.get("tags", {})
                    for k in tags.keys():
                        person_fieldnames.add(f"tags_{k}")

                # 确保关键字段排在前面
                key_fields = ["person_id", "bank_account_number", "bank", "is_deleted", "deleted_at"]
                other_fields = sorted([f for f in person_fieldnames if f not in key_fields])
                person_fieldnames = key_fields + other_fields
            else:
                # 追加模式：从现有CSV读取header，保持字段顺序一致
                import csv as csv_module
                with open(persons_file, 'r', encoding='utf-8') as f:
                    reader = csv_module.reader(f)
                    person_fieldnames = next(reader)  # 读取header行

            # 写入模式：第一次写入header，后续追加
            mode = 'w' if is_first_save else 'a'
            with open(persons_file, mode, encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=person_fieldnames)
                if is_first_save:
                    writer.writeheader()

                for account in new_profiles:
                    person = account.get("person", {})
                    person_id = account.get("account_id", "")

                    row = {
                        "person_id": person_id,
                        "bank_account_number": account.get("bank_account_number", ""),
                        "bank": account.get("bank", ""),
                        "is_deleted": account.get("is_deleted", False),
                        "deleted_at": account.get("deleted_at", "")
                    }

                    # 添加person字段（应用转换，跳过person_id）
                    for k, v in person.items():
                        if k == "person_id":
                            continue
                        if k in FIELD_DESCRIPTIONS and isinstance(v, int):
                            row[f"person_{k}"] = FIELD_DESCRIPTIONS[k].get(v, v)
                        else:
                            row[f"person_{k}"] = v

                    # 添加tags字段
                    tags = account.get("tags", {})
                    for k, v in tags.items():
                        if isinstance(v, list):
                            row[f"tags_{k}"] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[f"tags_{k}"] = v

                    writer.writerow(row)

                    # 标记为已保存
                    self.saved_person_ids.add(person_id)

            print(f"✓ 已{'保存' if is_first_save else '追加'}个人账户表: {persons_file} ({len(new_profiles)} 条)")


        # 2. 导出商户账户表（增量追加）
        merchants_file = os.path.join(self.output_dir, "merchants.csv")
        is_first_save_merchant = not os.path.exists(merchants_file)

        if new_merchants:
            # 收集所有字段 - 使用简洁的命名
            if is_first_save_merchant:
                # 第一次保存：收集所有字段
                merchant_fieldnames = set(["merchant_id", "bank_account_number", "bank", "is_deleted", "deleted_at"])
                for account in new_merchants:
                    merchant = account.get("merchant", {})
                    company = merchant.get("company", merchant)
                    if isinstance(company, dict):
                        for k in company.keys():
                            if k not in ["tags", "Company_id"]:
                                field_name = k.replace("Company_", "").replace("_", "_").lower()
                                if field_name not in ["id"]:
                                    merchant_fieldnames.add(field_name)
                    tags = account.get("tags", {})
                    for k in tags.keys():
                        merchant_fieldnames.add(f"tags_{k}")

                # 确保关键字段排在前面
                key_fields = ["merchant_id", "bank_account_number", "bank", "is_deleted", "deleted_at"]
                company_fields_order = ["description", "type", "registered_capital", "industry",
                                       "operating_status", "establishment_date", "legal_representative_id"]
                tags_fields = sorted([f for f in merchant_fieldnames if f.startswith('tags_')])

                company_fields = [f for f in company_fields_order if f in merchant_fieldnames]
                merchant_fieldnames = key_fields + company_fields + tags_fields
            else:
                # 追加模式：从现有CSV读取header，保持字段顺序一致
                import csv as csv_module
                with open(merchants_file, 'r', encoding='utf-8') as f:
                    reader = csv_module.reader(f)
                    merchant_fieldnames = next(reader)  # 读取header行

            # 写入模式：第一次写入header，后续追加
            mode = 'w' if is_first_save_merchant else 'a'
            with open(merchants_file, mode, encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=merchant_fieldnames)
                if is_first_save_merchant:
                    writer.writeheader()

                for account in new_merchants:
                    merchant = account.get("merchant", {})
                    merchant_id = account.get("account_id", "")

                    row = {
                        "merchant_id": merchant_id,
                        "bank_account_number": account.get("bank_account_number", ""),
                        "bank": account.get("bank", ""),
                        "is_deleted": account.get("is_deleted", False),
                        "deleted_at": account.get("deleted_at", "")
                    }

                    # 添加 company 字段
                    company = merchant.get("company", merchant)
                    if isinstance(company, dict):
                        field_mapping = {
                            "Company_description": "description",
                            "Company_type": "type",
                            "Registered_capital": "registered_capital",
                            "Industry": "industry",
                            "Operating_status": "operating_status",
                            "Establishment_date": "establishment_date",
                            "legal_representative_id": "legal_representative_id"
                        }
                        for orig_key, new_key in field_mapping.items():
                            if orig_key in company:
                                row[new_key] = company[orig_key]

                    # 添加 tags 字段
                    tags = account.get("tags", {})
                    for k, v in tags.items():
                        if isinstance(v, list):
                            row[f"tags_{k}"] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[f"tags_{k}"] = v

                    writer.writerow(row)

                    # 标记为已保存
                    self.saved_merchant_ids.add(merchant_id)

            print(f"✓ 已{'保存' if is_first_save_merchant else '追加'}商户账户表: {merchants_file} ({len(new_merchants)} 条)")


        # 3. 导出账户映射表（增量追加）
        mapping_file = os.path.join(self.output_dir, "account_mapping.csv")
        is_first_save_mapping = not os.path.exists(mapping_file)

        if new_profiles or new_merchants:
            mode = 'w' if is_first_save_mapping else 'a'
            with open(mapping_file, mode, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if is_first_save_mapping:
                    writer.writerow(["account_id", "account_type", "entity_id", "entity_type"])

                # 个人账户映射
                for account in new_profiles:
                    person_id = account.get("account_id", "unknown")
                    writer.writerow([person_id, "person", person_id, "person"])

                # 商户账户映射
                for account in new_merchants:
                    merchant_id = account.get("account_id", "unknown")
                    writer.writerow([merchant_id, "merchant", merchant_id, "company"])

            total_new = len(new_profiles) + len(new_merchants)
            print(f"✓ 已{'保存' if is_first_save_mapping else '追加'}账户映射表: {mapping_file} ({total_new} 条)")


    # ==================== 预算分配接口 ====================

    def allocate_budget(self, window_start: datetime, window_end: datetime, total_events: int) -> Dict[int, int]:
        """
        按 geo 分配交易预算

        Args:
            window_start: 时间窗口开始
            window_end: 时间窗口结束
            total_events: 总交易数

        Returns:
            geo_id -> 分配的交易数
        """
        # 按 geo 内个人数量分配
        geo_weights = {}
        total_persons = 0

        for geo_id in range(self.num_geos):
            num_persons = len(self.geo_persons.get(geo_id, []))
            geo_weights[geo_id] = num_persons
            total_persons += num_persons

        if total_persons == 0:
            # 降级：均匀分配
            per_geo = total_events // max(1, self.num_geos)
            return {geo_id: per_geo for geo_id in range(self.num_geos)}

        # 按比例分配
        allocation = {}
        allocated = 0

        for geo_id in range(self.num_geos):
            weight = geo_weights.get(geo_id, 0)
            geo_events = int(total_events * weight / total_persons)
            allocation[geo_id] = geo_events
            allocated += geo_events

        # 处理余数
        remainder = total_events - allocated
        if remainder > 0:
            # 分配给最大的 geo
            max_geo = max(geo_weights.keys(), key=lambda g: geo_weights[g])
            allocation[max_geo] += remainder

        return allocation

    def apply_events(self, events: List[Dict[str, Any]]) -> int:
        """
        应用（落地）交易事件

        Args:
            events: 交易事件列表，每项为包含交易字段的 dict

        Returns:
            成功添加的交易数
        """
        # 校验事件格式
        valid_events = []
        for event in events:
            if not isinstance(event, dict):
                continue

            # 必需字段
            required_fields = [
                "Timestamp", "From Bank", "From Account",
                "To Bank", "To Account", "Amount Received",
                "Receiving Currency", "Amount Paid",
                "Payment Currency", "Payment Format"
            ]

            if all(field in event for field in required_fields):
                valid_events.append(event)

        # 调用 add_transactions
        return self.add_transactions(valid_events)

    def _calculate_account_transaction_counts(self) -> Dict[str, int]:
        """
        计算每个账户的交易数量

        对于商户：使用接收交易数（txn_count_as_to）
        对于个人：使用发起交易数（txn_count_as_from）

        Returns:
            account_id -> 交易数量的字典
        """
        txn_counts = {}

        # 从账户数据结构中读取交易计数
        for acc_id, account in self.account_map.items():
            if account["account_type"] == "merchant":
                # 商户主要作为接收方，统计接收交易数
                txn_counts[acc_id] = account.get("txn_count_as_to", 0)
            else:
                # 个人主要作为发起方，统计发起交易数
                txn_counts[acc_id] = account.get("txn_count_as_from", 0)

        return txn_counts

    def remove_inactive_accounts(self, removal_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
        """
        删除交易数量最少的账户（按比例）

        Args:
            removal_ratio: 删除比例（默认10%）

        Returns:
            (removed_person_ids, removed_merchant_ids): 被删除的个人和商户账户ID列表
        """
        # 计算每个账户的交易数量
        txn_counts = self._calculate_account_transaction_counts()

        # 分别处理个人和商户账户
        person_counts = []
        merchant_counts = []

        for acc_id in self.person_ids:
            count = txn_counts.get(acc_id, 0)
            person_counts.append((acc_id, count))

        for acc_id in self.merchant_ids:
            count = txn_counts.get(acc_id, 0)
            merchant_counts.append((acc_id, count))

        # 按交易数量排序（升序）
        person_counts.sort(key=lambda x: x[1])
        merchant_counts.sort(key=lambda x: x[1])

        # 计算要删除的数量
        num_persons_to_remove = int(len(person_counts) * removal_ratio)
        num_merchants_to_remove = int(len(merchant_counts) * removal_ratio)

        # 获取要删除的账户ID
        removed_person_ids = [acc_id for acc_id, _ in person_counts[:num_persons_to_remove]]
        removed_merchant_ids = [acc_id for acc_id, _ in merchant_counts[:num_merchants_to_remove]]

        # 执行删除
        self._remove_accounts_from_system(removed_person_ids + removed_merchant_ids)

        print(f"[Account Churn] Removed {len(removed_person_ids)} persons and {len(removed_merchant_ids)} merchants")

        return removed_person_ids, removed_merchant_ids

    def _remove_accounts_from_system(self, account_ids: List[str]):
        """
        从系统中删除指定的账户（软删除：标记为已删除，但保留在accounts列表中）

        Args:
            account_ids: 要删除的账户ID列表
        """
        for acc_id in account_ids:
            account = self.account_map.get(acc_id)
            if not account:
                continue

            # 标记账户为已删除（软删除）
            account["is_deleted"] = True
            account["deleted_at"] = self.total_txn_generated  # 记录删除时的交易数

            # 从 account_map 中删除（但保留在 accounts 列表中）
            del self.account_map[acc_id]

            # 注意：不再从 self.accounts 列表中删除，保留用于最终导出

            # 从 person_ids 或 merchant_ids 中删除
            if account["account_type"] == "person":
                if acc_id in self.person_ids:
                    self.person_ids.remove(acc_id)
            elif account["account_type"] == "merchant":
                if acc_id in self.merchant_ids:
                    self.merchant_ids.remove(acc_id)

            # 从 geo 结构中删除
            geo_id = account.get("geo_id", 0)
            if account["account_type"] == "person":
                if geo_id in self.geo_persons and acc_id in self.geo_persons[geo_id]:
                    self.geo_persons[geo_id].remove(acc_id)
            elif account["account_type"] == "merchant":
                if geo_id in self.geo_merchants and acc_id in self.geo_merchants[geo_id]:
                    self.geo_merchants[geo_id].remove(acc_id)
                if geo_id in self.geo_merchant_popularity and acc_id in self.geo_merchant_popularity[geo_id]:
                    del self.geo_merchant_popularity[geo_id][acc_id]

            # 从 favorites 中删除
            if acc_id in self.local_favorites:
                del self.local_favorites[acc_id]
            if acc_id in self.external_favorites:
                del self.external_favorites[acc_id]

            # 从其他账户的 favorites 中删除该账户
            for fav_dict in [self.local_favorites, self.external_favorites]:
                for other_acc_id in list(fav_dict.keys()):
                    fav_dict[other_acc_id] = [(fav_id, weight) for fav_id, weight in fav_dict[other_acc_id] if fav_id != acc_id]

    async def add_new_accounts_async(self, num_persons: int, num_merchants: int, max_parallel: int = 10) -> Tuple[List[str], List[str]]:
        """
        异步生成并添加新账户到系统中

        Args:
            num_persons: 要添加的个人账户数量
            num_merchants: 要添加的商户账户数量
            max_parallel: 最大并行数

        Returns:
            (new_person_ids, new_merchant_ids): 新添加的个人和商户账户ID列表
        """
        from LLMGraph.utils.data_generator import TransactionDataGenerator

        new_person_ids = []
        new_merchant_ids = []

        # 计算起始ID（基于现有最大ID）
        person_start_id = self._get_next_person_id()
        merchant_start_id = self._get_next_merchant_id()

        # 生成个人账户
        if num_persons > 0:
            print(f"[Account Churn] Generating {num_persons} new persons (async)...")
            person_generator = TransactionDataGenerator(model_config_name="profile_generator")
            new_persons = await self._generate_persons_parallel(
                person_generator, num_persons, person_start_id, max_parallel
            )

            # 添加到系统
            self._add_persons_to_system(new_persons)
            new_person_ids = [p["person"]["person_id"] for p in new_persons if "person" in p]

        # 生成商户账户
        if num_merchants > 0:
            print(f"[Account Churn] Generating {num_merchants} new merchants (async)...")
            merchant_generator = TransactionDataGenerator(model_config_name="merchant_generator")
            new_merchants = await self._generate_merchants_parallel(
                merchant_generator, num_merchants, merchant_start_id, self.person_ids, max_parallel
            )

            # 添加到系统
            self._add_merchants_to_system(new_merchants)
            new_merchant_ids = [m.get("merchant_id", m.get("company", {}).get("Company_id")) for m in new_merchants]

        print(f"[Account Churn] Added {len(new_person_ids)} persons and {len(new_merchant_ids)} merchants")
        print(f"[Account Churn] 新账户将在当天交易结束后保存到CSV（确保有交易行为）")

        return new_person_ids, new_merchant_ids

    def _get_next_person_id(self) -> int:
        """获取下一个可用的个人账户ID编号"""
        if not self.person_ids:
            return 0

        # 从现有person_ids中提取最大编号
        max_id = 0
        for pid in self.person_ids:
            if pid.startswith("p"):
                try:
                    num = int(pid[1:])
                    max_id = max(max_id, num)
                except ValueError:
                    continue

        return max_id + 1

    def _get_next_merchant_id(self) -> int:
        """获取下一个可用的商户账户ID编号"""
        if not self.merchant_ids:
            return 0

        # 从现有merchant_ids中提取最大编号
        max_id = 0
        for mid in self.merchant_ids:
            if mid.startswith("c"):
                try:
                    num = int(mid[1:])
                    max_id = max(max_id, num)
                except ValueError:
                    continue

        return max_id + 1

    def _get_merchant_id(self, merchant: Dict[str, Any]) -> str:
        """从商户数据中提取merchant_id"""
        if "company" in merchant:
            company = merchant.get("company", {})
            if isinstance(company, dict):
                return company.get("Company_id", "")
        return merchant.get("merchant_id", "")

    def _add_persons_to_system(self, persons: List[Dict[str, Any]]):
        """将新生成的个人账户添加到系统中，并追加保存到 transaction_output"""
        person_count = len(self.accounts)

        for idx, item in enumerate(persons):
            if not isinstance(item, dict) or "person" not in item or "tags" not in item:
                continue

            person = item["person"]
            tags = item["tags"]
            person_id = person.get("person_id")

            if not person_id:
                continue

            account_idx = person_count + idx
            account = {
                "account_id": person_id,
                "bank_account_number": self._generate_bank_account_number(person_id, account_idx),
                "account_type": "person",
                "name": person.get("name", f"Person_{account_idx}"),
                "geo_id": person.get("geo_id", account_idx % self.num_geos),
                "balance": person.get("balance", 10000.0),
                "bank": random.choice(self.banks),
                "person": person,
                "tags": tags,
                "txn_count_as_from": 0,
                "txn_count_as_to": 0
            }

            self.accounts.append(account)
            self.account_map[person_id] = account
            self.person_ids.append(person_id)

            geo_id = account["geo_id"]
            if geo_id not in self.geo_persons:
                self.geo_persons[geo_id] = []
            self.geo_persons[geo_id].append(person_id)

            self.local_favorites[person_id] = []
            self.external_favorites[person_id] = []

        # 立即追加保存到 transaction_output/profiles_backup.json
        if persons:
            backup_file = os.path.join(self.output_dir, "profiles_backup.json")
            self._append_to_json_backup(backup_file, persons)

    def _add_merchants_to_system(self, merchants: List[Dict[str, Any]]):
        """将新生成的商户账户添加到系统中"""
        account_count = len(self.accounts)

        for idx, merchant in enumerate(merchants):
            if not isinstance(merchant, dict):
                continue

            # 支持两种格式
            if "company" in merchant:
                company = merchant["company"]
                merchant_id = company.get("Company_id")
                merchant_name = company.get("Company_name", f"Merchant_{idx}")
                tags = merchant.get("tags", {})
            else:
                merchant_id = merchant.get("merchant_id")
                merchant_name = merchant.get("merchant_name", f"Merchant_{idx}")
                tags = merchant.get("tags", {})

            if not merchant_id:
                continue

            account_idx = account_count + idx
            account = {
                "account_id": merchant_id,
                "bank_account_number": self._generate_bank_account_number(merchant_id, account_idx),
                "account_type": "merchant",
                "name": merchant_name,
                "geo_id": merchant.get("geo_id", account_idx % self.num_geos),
                "balance": merchant.get("balance", 50000.0),
                "bank": random.choice(self.banks),
                "merchant": merchant,
                "tags": tags,
                "txn_count_as_from": 0,  # 作为发起方的交易计数
                "txn_count_as_to": 0     # 作为接收方的交易计数
            }

            self.accounts.append(account)
            self.account_map[merchant_id] = account
            self.merchant_ids.append(merchant_id)

            geo_id = account["geo_id"]
            if geo_id not in self.geo_merchants:
                self.geo_merchants[geo_id] = []
            self.geo_merchants[geo_id].append(merchant_id)

            if geo_id not in self.geo_merchant_popularity:
                self.geo_merchant_popularity[geo_id] = {}
            self.geo_merchant_popularity[geo_id][merchant_id] = 0.0

            self.local_favorites[merchant_id] = []
            self.external_favorites[merchant_id] = []

        # 立即追加保存到 transaction_output/merchants_backup.json
        if merchants:
            backup_file = os.path.join(self.output_dir, "merchants_backup.json")
            self._append_to_json_backup(backup_file, merchants)

    def _append_to_json_backup(self, backup_file: str, records: List[Dict[str, Any]]):
        """追加记录到 JSON 备份文件

        Args:
            backup_file: 备份文件路径
            records: 要追加的记录列表
        """
        # 读取现有数据
        if os.path.exists(backup_file):
            with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "records" in data:
                    existing_records = data["records"]
                elif isinstance(data, list):
                    existing_records = data
                else:
                    existing_records = []
        else:
            existing_records = []

        # 追加新记录
        existing_records.extend(records)

        # 保存回文件
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump({"records": existing_records}, f, ensure_ascii=False, indent=2)

        print(f"✓ 已追加 {len(records)} 条记录到 {os.path.basename(backup_file)}")

    def _update_backup_files_with_bank_info(self, profiles: List[Dict], merchants: List[Dict]):
        """将 bank_account_number 和 bank 信息更新到备份文件

        Args:
            profiles: 个人账户列表
            merchants: 商户账户列表
        """
        # 更新个人账户的银行信息
        for item in profiles:
            if isinstance(item, dict) and "person" in item:
                person_id = item["person"].get("person_id")
                if person_id and person_id in self.account_map:
                    account = self.account_map[person_id]
                    item["bank_account_number"] = account.get("bank_account_number", "")
                    item["bank"] = account.get("bank", "")

        # 更新商户账户的银行信息
        for item in merchants:
            if isinstance(item, dict):
                merchant_id = self._get_merchant_id(item)
                if merchant_id and merchant_id in self.account_map:
                    account = self.account_map[merchant_id]
                    item["bank_account_number"] = account.get("bank_account_number", "")
                    item["bank"] = account.get("bank", "")

        # 保存更新后的数据到备份文件
        profiles_backup = os.path.join(self.output_dir, "profiles_backup.json")
        merchants_backup = os.path.join(self.output_dir, "merchants_backup.json")

        if profiles and os.path.exists(profiles_backup):
            with open(profiles_backup, "w", encoding="utf-8") as f:
                json.dump({"records": profiles}, f, ensure_ascii=False, indent=2)
            print(f"✓ 已更新个人账户银行信息到备份文件")

        if merchants and os.path.exists(merchants_backup):
            with open(merchants_backup, "w", encoding="utf-8") as f:
                json.dump({"records": merchants}, f, ensure_ascii=False, indent=2)
            print(f"✓ 已更新商户账户银行信息到备份文件")
