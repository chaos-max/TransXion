"""
场景日历解析器
用于解析和匹配基于日历的交易场景
"""
import os
import yaml
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import random


class ScenarioCalendar:
    """场景日历管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化场景日历

        Args:
            config_path: 配置文件路径，默认使用 scenario_calendar.yaml
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../tasks/transaction/scenario_calendar.yaml"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.fixed_scenarios = self.config.get('fixed_date_scenarios', [])
        self.lunar_scenarios = self.config.get('lunar_calendar_scenarios', [])
        self.periodic_scenarios = self.config.get('periodic_scenarios', [])
        self.ecommerce_scenarios = self.config.get('ecommerce_scenarios', [])
        self.seasonal_scenarios = self.config.get('seasonal_scenarios', [])

    def get_scenarios_for_date(self, target_date: datetime) -> List[Dict[str, Any]]:
        """
        获取指定日期的所有匹配场景

        Args:
            target_date: 目标日期

        Returns:
            匹配的场景列表
        """
        matched_scenarios = []

        # 检查固定日期场景
        matched_scenarios.extend(self._match_fixed_scenarios(target_date))

        # 检查周期性场景
        matched_scenarios.extend(self._match_periodic_scenarios(target_date))

        # 检查电商促销场景
        matched_scenarios.extend(self._match_ecommerce_scenarios(target_date))

        # 检查季节性场景
        matched_scenarios.extend(self._match_seasonal_scenarios(target_date))

        return matched_scenarios

    def _match_fixed_scenarios(self, target_date: datetime) -> List[Dict[str, Any]]:
        """匹配固定日期场景"""
        matched = []
        date_str = target_date.strftime("%m-%d")

        for scenario in self.fixed_scenarios:
            if date_str in scenario.get('dates', []):
                matched.append(scenario)

        return matched

    def _match_periodic_scenarios(self, target_date: datetime) -> List[Dict[str, Any]]:
        """匹配周期性场景"""
        matched = []

        for scenario in self.periodic_scenarios:
            pattern = scenario.get('pattern')

            if pattern == 'weekly':
                weekday = target_date.weekday()
                if weekday in scenario.get('weekdays', []):
                    matched.append(scenario)

            elif pattern == 'monthly':
                day = target_date.day
                if day in scenario.get('dates', []):
                    matched.append(scenario)

        return matched

    def _match_ecommerce_scenarios(self, target_date: datetime) -> List[Dict[str, Any]]:
        """匹配电商促销场景"""
        matched = []
        date_str = target_date.strftime("%m-%d")

        for scenario in self.ecommerce_scenarios:
            if date_str in scenario.get('dates', []):
                matched.append(scenario)

        return matched

    def _match_seasonal_scenarios(self, target_date: datetime) -> List[Dict[str, Any]]:
        """匹配季节性场景"""
        matched = []
        date_str = target_date.strftime("%m-%d")

        for scenario in self.seasonal_scenarios:
            date_range = scenario.get('date_range', [])
            if len(date_range) == 2:
                start_str, end_str = date_range
                if self._is_in_date_range(date_str, start_str, end_str):
                    matched.append(scenario)

        return matched

    @staticmethod
    def _is_in_date_range(date_str: str, start_str: str, end_str: str) -> bool:
        """判断日期是否在范围内"""
        return start_str <= date_str <= end_str

    def get_primary_scenario(self, target_date: datetime) -> Dict[str, Any]:
        """
        获取指定日期的主要场景（优先级最高的场景）

        Args:
            target_date: 目标日期

        Returns:
            主要场景信息
        """
        scenarios = self.get_scenarios_for_date(target_date)

        if not scenarios:
            return {
                'name': 'normal',
                'display_name': '正常日',
                'volume_multiplier': [1.0, 1.0],
                'description': '正常交易日',
                'categories': ['normal']
            }

        # 按优先级排序（volume_multiplier 越高优先级越高）
        scenarios.sort(key=lambda s: max(s.get('volume_multiplier', [1.0, 1.0])), reverse=True)

        return scenarios[0]

    def get_volume_multiplier(self, target_date: datetime) -> float:
        """
        获取指定日期的交易量乘数

        Args:
            target_date: 目标日期

        Returns:
            交易量乘数
        """
        scenario = self.get_primary_scenario(target_date)
        multiplier_range = scenario.get('volume_multiplier', [1.0, 1.0])

        # 在范围内随机选择
        return random.uniform(multiplier_range[0], multiplier_range[1])

    def get_scenario_summary(self, target_date: datetime) -> str:
        """
        获取指定日期的场景摘要描述

        Args:
            target_date: 目标日期

        Returns:
            场景摘要字符串
        """
        scenarios = self.get_scenarios_for_date(target_date)

        if not scenarios:
            return "正常交易日"

        primary = scenarios[0]
        summary = f"{primary['display_name']}: {primary['description']}"

        if len(scenarios) > 1:
            other_names = [s['display_name'] for s in scenarios[1:]]
            summary += f" (同时: {', '.join(other_names)})"

        return summary
