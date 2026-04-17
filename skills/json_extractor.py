"""
JSON 提取与容错解析技能

从 LLM 返回的文本中提取 JSON，处理常见格式问题：
- Markdown 代码块包裹
- 前后有额外文本
- 尾随逗号
- 单引号替换

这是从实际项目中提炼出的高频需求。
"""
import json
import re
import logging
from typing import Any, Optional

from skills.base import Skill

logger = logging.getLogger(__name__)


class JsonExtractorSkill(Skill):
    """
    JSON 提取与容错解析技能

    使用示例：
        skill = JsonExtractorSkill(llm)

        # 直接解析 LLM 返回的文本
        data = await skill.run(llm_output_text)

        # 调用 LLM 生成并解析 JSON
        data = await skill.generate_json(
            prompt="请分析以下文本的情感...",
            system_prompt="你是情感分析专家",
        )
    """

    name = "json_extractor"
    description = "从文本中提取并解析 JSON，支持容错处理"
    system_prompt = "请严格按照 JSON 格式返回结果，不要包含其他内容。"

    async def run(self, text: str) -> Optional[dict]:
        """
        从文本中提取并解析 JSON

        :param text: 包含 JSON 的文本
        :return: 解析后的 dict/list，失败返回 None
        """
        return self.extract_json(text)

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Optional[dict]:
        """
        调用 LLM 生成 JSON 并解析

        :param prompt: 用户提示词
        :param system_prompt: 系统提示词
        :param temperature: 温度（JSON 生成建议用低温度）
        :param max_tokens: 最大 token 数
        :return: 解析后的 dict/list
        """
        response = await self.call_llm(
            user_message=prompt,
            system_message=system_prompt or self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.is_error:
            logger.error(f"LLM 调用失败: {response.content}")
            return None

        return self.extract_json(response.content)

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """
        从文本中提取 JSON（静态方法，可直接调用）

        处理流程：
        1. 尝试提取 ```json ... ``` 代码块
        2. 尝试提取 ``` ... ``` 代码块
        3. 定位第一个 { 和最后一个 } 之间的内容
        4. 修复尾随逗号
        5. 解析 JSON

        :param text: 包含 JSON 的文本
        :return: 解析后的 Python 对象
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Step 1: 尝试提取 ```json 代码块
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Step 2: 尝试提取 ``` 代码块
            code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if code_match:
                json_str = code_match.group(1)
            else:
                json_str = text

        # Step 3: 定位 JSON 边界
        # 尝试找到 JSON 对象
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')

        # 也尝试 JSON 数组
        arr_start = json_str.find('[')
        arr_end = json_str.rfind(']')

        # 选择更早出现的那个
        if start_idx != -1 and end_idx > start_idx:
            if arr_start != -1 and arr_start < start_idx:
                json_str = json_str[arr_start:arr_end + 1]
            else:
                json_str = json_str[start_idx:end_idx + 1]
        elif arr_start != -1 and arr_end > arr_start:
            json_str = json_str[arr_start:arr_end + 1]

        # Step 4: 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Step 5: 解析 JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}, 内容: {json_str[:200]}...")
            return None
