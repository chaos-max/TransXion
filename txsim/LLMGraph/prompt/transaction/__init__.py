"""
Transaction scenario prompt templates
"""
from typing import Dict, Any
from datetime import datetime
from LLMGraph.prompt.base import BaseChatPromptTemplate

# 商户生成提示模板
MERCHANT_GENERATION_PROMPT = """You are a data generator for creating realistic merchant (company) profiles for a financial transaction simulation system.

Generate a merchant profile with the following fields:
- Company_id: A unique identifier in format "c{{number:06d}}" (e.g., "c000061")
- Company_description: A brief description combining company type (Small/Medium/Large), industry, and service/goods type
- Company_type: Choose ONE of "Small", "Medium", "Large" based on business nature:
  * Small: Street vendors, small shops, cafes, food stalls, convenience stores, small service providers
  * Medium: Chain stores, restaurants, clinics, medium-sized retailers, professional services
  * Large: Wholesale businesses, manufacturers, large enterprises, B2B service providers, major retailers
  IMPORTANT: The type MUST match the business description (e.g., a food stall cannot be "Large")
- Registered_capital: Capital amount in CNY (Small: 1M-5M, Medium: 5M-50M, Large: 50M-200M)
- Industry: One of {industries}
- Operating_status: One of "Active", "Suspended", "Under Review"
- Establishment_date: Date in format "YYYY-MM-DD"
- legal_representative_id: A person ID that exists in the system

Also generate transaction behavior tags:
- avg_txn_cnt_daily: Average daily transaction count (reasonable for company size)
- avg_txn_amt: Average transaction amount in CNY (MUST follow these ranges based on company type):
  * Small merchants: 50-200 CNY (daily small purchases, street food, convenience stores)
  * Medium merchants: 500-3000 CNY (moderate business transactions, restaurants, services)
  * Large merchants: 5000-20000 CNY (large-scale transactions, wholesale, enterprise services)
  * Choose a value within the appropriate range that fits the industry and business model
- tod_p: MUST be an array of EXACTLY 8 float numbers representing time-of-day probability distribution
  * Each value represents a 3-hour period: [0:00-3:00, 3:00-6:00, 6:00-9:00, 9:00-12:00, 12:00-15:00, 15:00-18:00, 18:00-21:00, 21:00-24:00]
  * All 8 values MUST sum to 1.0
  * Example: [0.02, 0.02, 0.08, 0.20, 0.25, 0.23, 0.15, 0.05]

Context:
{context}

IMPORTANT: Output ONLY a valid JSON object with "company" and "tags" fields. Do NOT add any explanatory text, code blocks, or markdown formatting.

Example output format:
{{
  "company": {{
    "Company_id": "c000061",
    "Company_description": "Small retail company specializing in consumer electronics",
    "Company_type": "Small",
    "Registered_capital": 2500000.0,
    "Industry": "Retail",
    "Operating_status": "Active",
    "Establishment_date": "2020-03-15",
    "legal_representative_id": "p001234"
  }},
  "tags": {{
    "avg_txn_cnt_daily": 85.0,
    "avg_txn_amt": 250.5,
    "tod_p": [0.02, 0.02, 0.08, 0.20, 0.25, 0.23, 0.15, 0.05]
  }}
}}
"""

# 个人生成提示模板
PERSON_GENERATION_PROMPT = """You are a data generator for creating realistic person profiles for a financial transaction simulation system.

Generate {num_persons} diverse person profiles. Each profile should have the following fields:
- person_id: A unique identifier in format "p{{number:06d}}" (e.g., "p002001", "p002002", ...)
- age: Age between 18 and 80
- gender: 0 for male, 1 for female
- occupation: Integer code 0-9 representing different occupations
- marital_status: 0 for single, 1 for married, 2 for divorced
- education: 0-4 representing education level (0: high school, 1: associate, 2: bachelor, 3: master, 4: doctorate)

Also generate transaction behavior tags for each person:
- avg_txn_cnt_daily: Average daily transaction count (typically 1-10 for individuals)
- avg_txn_amt: Average transaction amount (reasonable for occupation and lifestyle)
- tod_p: MUST be an array of EXACTLY 8 float numbers representing time-of-day probability distribution
  * Each value represents a 3-hour period: [0:00-3:00, 3:00-6:00, 6:00-9:00, 9:00-12:00, 12:00-15:00, 15:00-18:00, 18:00-21:00, 21:00-24:00]
  * All 8 values MUST sum to 1.0
  * Example: [0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.05]

Context:
{context}

IMPORTANT:
- Output ONLY a valid JSON array containing {num_persons} person objects
- Do NOT add any explanatory text, code blocks, or markdown formatting
- Ensure diversity in age, gender, occupation, marital status, and education levels
- Each person should have realistic and varied transaction behavior patterns

Example output format for {num_persons} persons:
[
  {{
    "person": {{
      "person_id": "p001000",
      "age": 35,
      "gender": 1,
      "occupation": 3,
      "marital_status": 1,
      "education": 2
    }},
    "tags": {{
      "avg_txn_cnt_daily": 4.5,
      "avg_txn_amt": 85.0,
      "tod_p": [0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.05]
    }}
  }},
  {{
    "person": {{
      "person_id": "p001001",
      "age": 28,
      "gender": 0,
      "occupation": 5,
      "marital_status": 0,
      "education": 2
    }},
    "tags": {{
      "avg_txn_cnt_daily": 6.2,
      "avg_txn_amt": 120.0,
      "tod_p": [0.03, 0.02, 0.08, 0.18, 0.22, 0.28, 0.15, 0.04]
    }}
  }}
]
"""


class MerchantGenerationPromptTemplate(BaseChatPromptTemplate):
    """商户生成Prompt模板"""
    def __init__(self, **kwargs):
        super().__init__(
            template=MERCHANT_GENERATION_PROMPT,
            input_variables=["industries", "context"],
            **kwargs
        )


class PersonGenerationPromptTemplate(BaseChatPromptTemplate):
    """个人生成Prompt模板"""
    def __init__(self, **kwargs):
        super().__init__(
            template=PERSON_GENERATION_PROMPT,
            input_variables=["num_persons", "context"],
            **kwargs
        )


# 交易生成规划提示模板
TRANSACTION_PLANNING_PROMPT_TEMPLATE = """You are an expert transaction behavior planner for a financial simulation system. Your goal is to generate realistic, diverse transaction patterns that reflect real-world consumer behavior.

**Time Window**: {time_window}

**Geographic Region Summary** (geo_id={geo_id}):
- Number of persons: {n_persons:,}
- Number of merchants: {n_merchants:,}
- Target transactions: {target_txn_count:,}{top_merchants_section}{recent_stats_section}{promotion_section}

**Your Task**:
Analyze the above context and output a JSON plan that controls transaction generation distribution. Consider:
1. Time of day patterns (business hours vs. evening)
2. Transaction size distribution (small daily purchases vs. large occasional purchases)
3. Payment method preferences (digital vs. traditional)
4. Cross-region transaction likelihood
5. Merchant type preferences (if applicable)

**Output Format**:
Output ONLY a single JSON object (no markdown code blocks, no explanations, no additional text) with this exact structure:

{{
  "cross_delta": 0.02,
  "amount_buckets": [
    {{"name": "small", "p": 0.5, "mult_range": [0.3, 0.8]}},
    {{"name": "mid", "p": 0.35, "mult_range": [0.8, 1.5]}},
    {{"name": "large", "p": 0.15, "mult_range": [1.5, 3.0]}}
  ],
  "payment_format_p": {{"Mobile": 0.4, "Card": 0.3, "Transfer": 0.2, "Cash": 0.1}},
  "merchant_type_bias": {{}}
}}

**Field Specifications**:

1. **cross_delta** (float, range: [-0.05, 0.05]):
   - Adjustment to the base cross-region transaction probability
   - Positive values increase cross-region transactions
   - Negative values decrease cross-region transactions
   - Example: 0.02 means +2% more cross-region transactions

2. **amount_buckets** (list of 2-4 objects):
   - Each bucket represents a transaction size category
   - Fields per bucket:
     - "name": Category name (e.g., "small", "mid", "large")
     - "p": Probability weight (all buckets must sum to 1.0)
     - "mult_range": [min, max] multiplier range for base transaction amount
   - Example: {{"name": "small", "p": 0.5, "mult_range": [0.3, 0.8]}}
     means 50% of transactions are small, with amounts 0.3x-0.8x the base

3. **payment_format_p** (object):
   - Probability distribution over payment formats
   - Valid formats: "Mobile", "Card", "Transfer", "Cash"
   - All probabilities must sum to 1.0
   - Example: {{"Mobile": 0.4, "Card": 0.3, "Transfer": 0.2, "Cash": 0.1}}

4. **merchant_type_bias** (object, optional):
   - Preference weights for merchant types (if type information is available)
   - Keys: merchant type names (e.g., "restaurant", "retail", "grocery")
   - Values: preference weights (higher = more preferred)
   - Use empty object {{}} if no type information is available
   - Example: {{"restaurant": 0.4, "retail": 0.3, "grocery": 0.3}}

**Important Constraints**:
- All probability distributions MUST sum to 1.0 (tolerance: ±0.001)
- cross_delta MUST be in range [-0.05, 0.05]
- amount_buckets MUST have 2-4 entries
- mult_range values MUST be positive and min < max
- Output MUST be valid JSON (no trailing commas, proper quotes)
- Do NOT include any explanatory text before or after the JSON

Generate a realistic plan now:"""


def build_transaction_planning_prompt(
    geo_id: int,
    window_start: datetime,
    window_end: datetime,
    geo_summary: Dict[str, Any]
) -> str:
    """
    构造交易生成规划的 LLM prompt

    Args:
        geo_id: 地理区域 ID
        window_start: 时间窗口开始
        window_end: 时间窗口结束
        geo_summary: 地理区域摘要信息

    Returns:
        格式化的 prompt 字符串
    """
    # 提取基础信息
    n_persons = geo_summary.get('n_persons', 0)
    n_merchants = geo_summary.get('n_merchants', 0)
    target_txn_count = geo_summary.get('target_txn_count', 0)

    # 构造时间窗口描述
    time_window = f"{window_start.strftime('%Y-%m-%d %H:%M:%S')} to {window_end.strftime('%Y-%m-%d %H:%M:%S')}"

    # 构造 top_merchants 描述
    top_merchants = geo_summary.get('top_merchants', [])
    top_merchants_section = ""
    if top_merchants:
        merchant_list = ", ".join([f"{m[0]} ({m[1]} txns)" for m in top_merchants[:5]])
        top_merchants_section = f"\n- Top merchants: {merchant_list}"

    # 构造 recent_stats 描述
    recent_stats = geo_summary.get('recent_stats', {})
    recent_stats_section = ""
    if recent_stats:
        p2m_ratio = recent_stats.get('p2m_ratio', 0.75)
        cross_bank_ratio = recent_stats.get('cross_bank_ratio', 0.1)
        format_dist = recent_stats.get('format_distribution', {})

        recent_stats_section = f"\n- Recent P2M ratio: {p2m_ratio:.2%}"
        recent_stats_section += f"\n- Recent cross-bank ratio: {cross_bank_ratio:.2%}"

        if format_dist:
            format_str = ", ".join([f"{k}: {v:.2%}" for k, v in format_dist.items()])
            recent_stats_section += f"\n- Recent payment format distribution: {format_str}"

    # 构造 promotion_theme 描述
    promotion_theme = geo_summary.get('promotion_theme')
    promotion_section = ""
    if promotion_theme:
        promotion_section = f"\n- Promotion theme: {promotion_theme}"

    # 使用模板格式化 prompt
    return TRANSACTION_PLANNING_PROMPT_TEMPLATE.format(
        time_window=time_window,
        geo_id=geo_id,
        n_persons=n_persons,
        n_merchants=n_merchants,
        target_txn_count=target_txn_count,
        top_merchants_section=top_merchants_section,
        recent_stats_section=recent_stats_section,
        promotion_section=promotion_section
    )


# 日级场景生成提示模板
DAILY_SCENARIO_SYSTEM_PROMPT = """You are a daily scenario director for a China mainland transaction simulation system. Your task is to generate realistic daily variations in transaction volume, geographic distribution, and promotional themes based on Chinese e-commerce and payment patterns.

CRITICAL: You MUST output ONLY a valid JSON object. Do NOT include:
- Any explanatory text before or after the JSON
- Code blocks or markdown formatting (no ```json or ```)
- Comments or additional information
- Just the raw JSON object starting with { and ending with }"""

DAILY_SCENARIO_USER_PROMPT_TEMPLATE = """Generate a daily scenario for China mainland transaction simulation.

Context:
- This simulates transactions in mainland China across 20 geographic regions (geo_id: 0-19)
- Geographic regions represent different Chinese cities/provinces
- Consider Chinese shopping festivals, holidays, and payment patterns

Current Statistics:
{stats_str}

Output a JSON object with the following schema:
{{
  "volume_multiplier": <float between 0.5 and 2.0>,
  "geo_multipliers": {{"0": <float>, "5": <float>, "10": <float>, ...}},
  "promotion_theme": "<one of: double11, 618, spring_festival, payday, normal>"
}}

Requirements:
- volume_multiplier: Overall daily transaction volume (0.5 = 50% of baseline, 2.0 = 200%)
- geo_multipliers: Adjust specific regions by geo_id (0-19). Only specify regions that differ from baseline (1.0)
  * Example: {{"0": 1.5, "5": 1.3, "10": 0.8}} means region 0 has 150%, region 5 has 130%, region 10 has 80%
  * Omit regions that should use default 1.0 multiplier
- promotion_theme options (choose the most appropriate one):
  * "double11": Double 11 shopping festival (Nov 10-12) - highest volume
  * "618_festival": 618 mid-year shopping festival (Jun 16-18) - very high volume
  * "double12": Double 12 shopping festival (Dec 12) - high volume
  * "spring_festival": Chinese New Year period - very high volume, family/gift focus
  * "new_year_day": New Year's Day (Jan 1) - elevated volume
  * "valentines_day": Valentine's Day (Feb 14) - gift/dining focus
  * "womens_day": Women's Day (Mar 8) - beauty/fashion focus
  * "queen_festival": 38 Queen Festival (Mar 7-9) - women's shopping
  * "labor_day": Labor Day holiday (May 1-3) - travel/shopping
  * "childrens_day": Children's Day (Jun 1) - family/kids focus
  * "qixi_festival": Qixi Festival (Chinese Valentine's) - gift/dining focus
  * "mid_autumn_festival": Mid-Autumn Festival - gift/family focus
  * "national_day": National Day Golden Week (Oct 1-7) - travel/shopping peak
  * "99_sale": 99 Sale (Sep 8-10) - back to school promotion
  * "new_year_goods": New Year Goods Festival (mid-Jan) - holiday prep
  * "back_to_school": Back to school season (late Aug/early Sep) - education/electronics
  * "christmas": Christmas (Dec 24-25) - gift/dining focus
  * "payday": Month-end payday (25th-31st) - slightly elevated volume
  * "weekend": Weekend - leisure/shopping increase
  * "friday": Friday - weekend prep
  * "summer_sale": Summer promotion - seasonal goods
  * "winter_sale": Winter promotion - seasonal goods
  * "normal": Regular day - baseline volume
- Consider day of week: weekends typically have higher volume
- Consider date patterns: month-end (payday), major shopping festivals
- Create realistic variations based on Chinese consumer behavior

CRITICAL OUTPUT REQUIREMENTS:
1. Output ONLY the JSON object
2. Start directly with {{ (opening brace)
3. End directly with }} (closing brace)
4. NO markdown code blocks (no ```json or ```)
5. NO explanatory text before or after
6. NO comments inside the JSON

Example of correct output format:
{{"volume_multiplier": 1.0, "geo_multipliers": {{}}, "promotion_theme": "normal"}}"""


def build_daily_scenario_prompt(global_stats: Dict[str, Any]) -> tuple:
    """
    构造日级场景生成的 LLM prompt

    Args:
        global_stats: 全局统计信息
            - cur_date: 当前日期
            - total_txn_so_far: 已生成交易数（可选）
            - remaining_txn: 剩余交易数（可选）
            - recent_format_distribution: 最近格式分布（可选）
            - recent_cross_ratio: 最近跨币种比例（可选）
            - per_geo_recent_counts: 各 geo 最近交易数（可选）
            - top_merchants: 热门商户（可选）

    Returns:
        (system_prompt, user_prompt) 元组
    """
    import json

    # 格式化统计信息为 JSON 字符串
    stats_str = json.dumps(global_stats, indent=2, default=str)

    # 使用模板格式化用户 prompt
    user_prompt = DAILY_SCENARIO_USER_PROMPT_TEMPLATE.format(stats_str=stats_str)

    return DAILY_SCENARIO_SYSTEM_PROMPT, user_prompt

