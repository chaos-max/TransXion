"""
Transaction scenario output parsers
"""
import json
from typing import Union, Dict, Any
from LLMGraph.output_parser.base_parser import AgentOutputParser, find_and_load_json


class MerchantGenerationParser(AgentOutputParser):
    """商户生成结果解析器"""

    def parse(self, text: str) -> Union[Dict[str, Any], dict]:
        """
        解析LLM输出的商户数据

        Expected format:
        {
            "company": {
                "Company_id": "c000061",
                "Company_description": "...",
                "Company_type": "Small",
                "Registered_capital": 1500000.0,
                "Industry": "Retail",
                "Operating_status": "Active",
                "Establishment_date": "2020-01-15",
                "legal_representative_id": "p001234"
            },
            "tags": {
                "avg_txn_cnt_daily": 50.0,
                "avg_txn_amt": 100.5,
                "tod_p": [0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.1, 0.05]
            }
        }
        """
        try:
            # 尝试解析JSON
            output_json = find_and_load_json(text, "dict")

            # 验证必需字段
            if not isinstance(output_json, dict):
                return {"fail": True, "error": "Output is not a dictionary"}

            if "company" not in output_json or "tags" not in output_json:
                return {"fail": True, "error": "Missing 'company' or 'tags' field"}

            company = output_json["company"]
            tags = output_json["tags"]

            # 验证company字段
            required_company_fields = [
                "Company_id", "Company_description", "Company_type",
                "Registered_capital", "Industry", "Operating_status",
                "Establishment_date", "legal_representative_id"
            ]
            for field in required_company_fields:
                if field not in company:
                    return {"fail": True, "error": f"Missing company field: {field}"}

            # 验证tags字段
            required_tags_fields = ["avg_txn_cnt_daily", "avg_txn_amt", "tod_p"]
            for field in required_tags_fields:
                if field not in tags:
                    return {"fail": True, "error": f"Missing tags field: {field}"}

            # 验证并修复 tod_p 格式
            tod_p = tags.get("tod_p")

            # 如果 tod_p 不存在或格式错误，使用默认值
            if not tod_p:
                # 默认均匀分布
                tags["tod_p"] = [0.125] * 8
            elif not isinstance(tod_p, list):
                # 如果不是列表，尝试转换
                try:
                    if isinstance(tod_p, str):
                        # 可能是字符串形式的列表
                        import ast
                        tod_p = ast.literal_eval(tod_p)
                        tags["tod_p"] = tod_p
                    else:
                        # 无法转换，使用默认值
                        tags["tod_p"] = [0.125] * 8
                except (ValueError, TypeError, SyntaxError):
                    tags["tod_p"] = [0.125] * 8

            # 验证长度
            if len(tags["tod_p"]) != 8:
                # 长度不对，尝试修复
                if len(tags["tod_p"]) < 8:
                    # 补齐到8个
                    while len(tags["tod_p"]) < 8:
                        tags["tod_p"].append(0.125)
                else:
                    # 截断到8个
                    tags["tod_p"] = tags["tod_p"][:8]

            # 归一化tod_p (确保和为1)
            tod_sum = sum(tags["tod_p"])
            if abs(tod_sum) < 0.001:  # 避免除以0
                tags["tod_p"] = [0.125] * 8
            elif abs(tod_sum - 1.0) > 0.01:  # 允许小误差
                tags["tod_p"] = [p / tod_sum for p in tags["tod_p"]]

            return output_json

        except Exception as e:
            return {"fail": True, "error": str(e)}


class PersonGenerationParser(AgentOutputParser):
    """个人生成结果解析器（支持单个或批量解析）"""

    def parse(self, text: str) -> Union[Dict[str, Any], list, dict]:
        """
        解析LLM输出的个人数据（支持单个对象或数组）

        Expected format (single):
        {
            "person": {...},
            "tags": {...}
        }

        Expected format (batch):
        [
            {"person": {...}, "tags": {...}},
            {"person": {...}, "tags": {...}}
        ]
        """
        try:
            # 先尝试解析为列表（批量模式）
            output_json = find_and_load_json(text, "list")

            if isinstance(output_json, list):
                # 批量模式：验证每个元素
                validated_persons = []
                for idx, person_data in enumerate(output_json):
                    validated = self._validate_single_person(person_data)
                    # 检查是否验证失败（失败时返回包含"fail": True的字典）
                    if isinstance(validated, dict) and validated.get("fail"):
                        # 跳过无效数据
                        continue
                    validated_persons.append(validated)

                if validated_persons:
                    return validated_persons
                else:
                    return {"fail": True, "error": "No valid persons in batch"}

            # 如果find_and_load_json返回的是字符串，说明解析失败
            if isinstance(output_json, str):
                # 尝试修复不完整的JSON（可能被截断）
                import json
                try:
                    # 尝试直接解析
                    parsed = json.loads(output_json)
                    if isinstance(parsed, list):
                        # 递归调用自己处理列表
                        return self.parse(json.dumps(parsed))
                except json.JSONDecodeError:
                    return {"fail": True, "error": f"JSON parse failed, response may be truncated. Length: {len(output_json)}"}

            # 如果不是列表也不是字符串，尝试解析为单个对象
            output_json = find_and_load_json(text, "dict")
            return self._validate_single_person(output_json)

        except Exception as e:
            return {"fail": True, "error": f"Parse error: {str(e)}"}

    def _validate_single_person(self, output_json: dict) -> Union[Dict[str, Any], dict]:
        """验证单个个人数据对象"""
        try:
            # 验证必需字段
            if not isinstance(output_json, dict):
                return {"fail": True, "error": "Output is not a dictionary"}

            if "person" not in output_json or "tags" not in output_json:
                return {"fail": True, "error": "Missing 'person' or 'tags' field"}

            person = output_json["person"]
            tags = output_json["tags"]

            # 验证person字段
            required_person_fields = [
                "person_id", "age", "gender", "occupation",
                "marital_status", "education"
            ]
            for field in required_person_fields:
                if field not in person:
                    return {"fail": True, "error": f"Missing person field: {field}"}

            # 验证tags字段
            required_tags_fields = ["avg_txn_cnt_daily", "avg_txn_amt", "tod_p"]
            for field in required_tags_fields:
                if field not in tags:
                    return {"fail": True, "error": f"Missing tags field: {field}"}

            # 验证并修复 tod_p 格式
            tod_p = tags.get("tod_p")

            # 如果 tod_p 不存在或格式错误，使用默认值
            if not tod_p:
                # 默认均匀分布
                tags["tod_p"] = [0.125] * 8
            elif not isinstance(tod_p, list):
                # 如果不是列表，尝试转换
                try:
                    if isinstance(tod_p, str):
                        # 可能是字符串形式的列表
                        import ast
                        tod_p = ast.literal_eval(tod_p)
                        tags["tod_p"] = tod_p
                    else:
                        # 无法转换，使用默认值
                        tags["tod_p"] = [0.125] * 8
                except (ValueError, TypeError, SyntaxError):
                    tags["tod_p"] = [0.125] * 8

            # 验证长度
            if len(tags["tod_p"]) != 8:
                # 长度不对，尝试修复
                if len(tags["tod_p"]) < 8:
                    # 补齐到8个
                    while len(tags["tod_p"]) < 8:
                        tags["tod_p"].append(0.125)
                else:
                    # 截断到8个
                    tags["tod_p"] = tags["tod_p"][:8]

            # 归一化tod_p (确保和为1)
            tod_sum = sum(tags["tod_p"])
            if abs(tod_sum) < 0.001:  # 避免除以0
                tags["tod_p"] = [0.125] * 8
            elif abs(tod_sum - 1.0) > 0.01:  # 允许小误差
                tags["tod_p"] = [p / tod_sum for p in tags["tod_p"]]

            # 验证数值范围
            if not (18 <= person["age"] <= 80):
                return {"fail": True, "error": "age must be between 18 and 80"}

            if person["gender"] not in [0, 1]:
                return {"fail": True, "error": "gender must be 0 or 1"}

            if not (0 <= person["occupation"] <= 9):
                return {"fail": True, "error": "occupation must be between 0 and 9"}

            if person["marital_status"] not in [0, 1, 2]:
                return {"fail": True, "error": "marital_status must be 0, 1, or 2"}

            if not (0 <= person["education"] <= 4):
                return {"fail": True, "error": "education must be between 0 and 4"}

            return output_json

        except Exception as e:
            return {"fail": True, "error": str(e)}
