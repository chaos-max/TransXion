"""
LLM-based data generator for transaction profiles
支持按任务功能配置模型，自动识别本地/API模型并支持降级
"""
import json
import os
import hashlib
from typing import List, Dict, Any, Optional
from agentscope.models import ModelWrapperBase, load_model_by_config_name
from LLMGraph.prompt.transaction import MerchantGenerationPromptTemplate, PersonGenerationPromptTemplate
from LLMGraph.output_parser.transaction import MerchantGenerationParser, PersonGenerationParser


def generate_account_id(prefix: str, index: int) -> str:
    """
    生成账户ID（数字格式）

    Args:
        prefix: 前缀 ('p' for person, 'c' for company/merchant)
        index: 索引编号

    Returns:
        格式为 prefix + 6位数字 (例如: p000100, c000001)
    """
    # 使用6位数字格式，与现有账户ID格式保持一致
    return f"{prefix}{index:06d}"


class TransactionDataGenerator:
    """交易数据生成器（支持配置文件中的模型选择和自动降级）"""

    # 行业列表
    INDUSTRIES = [
        "Retail", "Food & Beverage", "Entertainment", "Healthcare",
        "Education", "Transportation", "E-commerce", "Utilities",
        "Telecommunications", "Financial Services", "Manufacturing",
        "Travel & Hospitality"
    ]

    def __init__(self, model_config_name: str = "profile_generator"):
        """
        初始化数据生成器

        Args:
            model_config_name: 模型配置名称（如 profile_generator, merchant_generator）
        """
        self.model_config_name = model_config_name
        self._model = None

        # 初始化prompt模板
        self.merchant_prompt = MerchantGenerationPromptTemplate()
        self.person_prompt = PersonGenerationPromptTemplate()

        # 初始化parser
        self.merchant_parser = MerchantGenerationParser()
        self.person_parser = PersonGenerationParser()

    def _get_model(self) -> ModelWrapperBase:
        """获取模型实例（支持 fallback_chain 自动降级）"""
        if self._model is not None:
            return self._model

        fallback_chain = self._get_fallback_chain(self.model_config_name)
        if not fallback_chain:
            fallback_chain = [self.model_config_name]

        errors = []
        for config_name in fallback_chain:
            try:
                model = load_model_by_config_name(config_name)
                test_response = model([{"role": "user", "content": "test"}])
                self._model = model
                return model
            except Exception as e:
                errors.append(f"{config_name}: {str(e)}")
                continue

        raise RuntimeError(
            f"无法加载模型配置 {self.model_config_name} 及其降级链: {fallback_chain}\n错误详情:\n" + "\n".join(errors)
        )

    def _get_fallback_chain(self, config_name: str) -> Optional[List[str]]:
        """
        获取配置的降级链

        Args:
            config_name: 配置名称

        Returns:
            降级链列表，如果没有则返回 None
        """
        try:
            # 读取配置文件
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../llms/default_model_configs.json"
            )

            with open(config_path, 'r') as f:
                data = json.load(f)

            # 获取 model_configs 列表
            model_configs = data.get("model_configs", [])

            # 查找当前配置
            for config in model_configs:
                if isinstance(config, dict) and config.get("config_name") == config_name:
                    return config.get("fallback_chain")

            return None
        except Exception:
            # 静默处理错误，避免干扰输出
            return None

    def generate_merchants(
        self,
        num_merchants: int,
        start_id: int,
        existing_person_ids: List[str],
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        生成商户数据

        Args:
            num_merchants: 需要生成的商户数量
            start_id: 起始ID编号
            existing_person_ids: 现有的person_id列表,用于legal_representative_id
            batch_size: 每批生成数量(目前仅支持1)

        Returns:
            商户数据列表
        """
        merchants = []

        for i in range(num_merchants):
            merchant_id = generate_account_id("c", start_id + i)

            # 构造上下文
            context = f"""
Generating merchant {i+1}/{num_merchants}.
Merchant ID: {merchant_id}
Available person IDs for legal representative: {len(existing_person_ids)} persons
Please select a random person ID from the existing list for legal_representative_id.
Make sure the generated data is realistic and consistent with the company type and industry.
"""

            # 准备prompt
            prompt_vars = {
                "industries": ", ".join(self.INDUSTRIES),
                "context": context
            }
            messages = self.merchant_prompt.format_messages(**prompt_vars)

            # 调用模型
            try:
                model = self._get_model()
                response = model(messages)
                parsed = self.merchant_parser.parse(response.text)

                if "fail" in parsed and parsed["fail"]:
                    continue

                # 强制覆盖ID和legal_representative_id
                parsed["company"]["Company_id"] = merchant_id
                if existing_person_ids:
                    import random
                    parsed["company"]["legal_representative_id"] = random.choice(existing_person_ids)

                merchants.append(parsed)

            except Exception as e:
                continue

        return merchants

    def generate_persons(
        self,
        num_persons: int,
        start_id: int,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成个人数据（支持批量生成）

        Args:
            num_persons: 需要生成的个人数量
            start_id: 起始ID编号
            batch_size: 每批生成数量（默认5，建议3-10之间）

        Returns:
            个人数据列表
        """
        persons = []

        # 按批次生成
        for batch_start in range(0, num_persons, batch_size):
            batch_end = min(batch_start + batch_size, num_persons)
            current_batch_size = batch_end - batch_start

            # 生成本批次的person_id列表
            person_ids = [
                generate_account_id("p", start_id + batch_start + i)
                for i in range(current_batch_size)
            ]

            # 构造上下文
            context = f"""
Generating batch {batch_start // batch_size + 1}, persons {batch_start + 1}-{batch_end} of {num_persons}.
Person IDs for this batch: {', '.join(person_ids)}
Please generate {current_batch_size} diverse and realistic person profiles.
Ensure variety in demographics (age, gender, occupation, education) and transaction behaviors.
"""

            # 准备prompt
            prompt_vars = {
                "num_persons": current_batch_size,
                "context": context
            }
            messages = self.person_prompt.format_messages(**prompt_vars)

            # 调用模型
            try:
                model = self._get_model()
                response = model(messages)

                # 调试：检查响应对象
                if not response.text or len(response.text) == 0:
                    print(f"  警告: 批次 {batch_start // batch_size + 1} API返回空响应")
                    print(f"  响应对象类型: {type(response)}")
                    print(f"  响应属性: {dir(response)}")
                    if hasattr(response, 'raw'):
                        print(f"  原始响应: {response.raw}")
                    if hasattr(response, 'error'):
                        print(f"  错误信息: {response.error}")
                    continue

                parsed = self.person_parser.parse(response.text)

                if isinstance(parsed, dict) and "fail" in parsed and parsed["fail"]:
                    print(f"  警告: 批次 {batch_start // batch_size + 1} 生成失败: {parsed.get('error', 'Unknown error')}")
                    print(f"  原始响应: {response.text[:500]}...")
                    continue

                # parsed 应该是一个列表
                if isinstance(parsed, list):
                    # 强制覆盖ID（确保ID正确）
                    for i, person_data in enumerate(parsed):
                        if i < len(person_ids):
                            person_data["person"]["person_id"] = person_ids[i]
                            persons.append(person_data)
                else:
                    print(f"  警告: 批次 {batch_start // batch_size + 1} 返回格式错误，期望list但得到{type(parsed)}")
                    print(f"  原始响应: {response.text[:500]}...")

            except Exception as e:
                print(f"  警告: 批次 {batch_start // batch_size + 1} 生成异常: {e}")
                import traceback
                traceback.print_exc()
                continue

        return persons
