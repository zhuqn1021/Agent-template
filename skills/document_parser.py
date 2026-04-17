"""
文档解析技能

利用 LLM 从非结构化文本中提取结构化信息。
"""
import logging
from typing import Optional

from skills.base import Skill
from skills.json_extractor import JsonExtractorSkill

logger = logging.getLogger(__name__)


class DocumentParserSkill(Skill):
    """
    文档解析技能

    使用 LLM 从文本中提取结构化数据，支持自定义提取 Schema。

    使用示例：
        skill = DocumentParserSkill(llm)

        # 使用默认 Schema 解析
        result = await skill.run("张三，男，1990年出生...")

        # 使用自定义 Schema 解析
        result = await skill.run(
            text="...",
            schema={
                "name": "姓名",
                "age": "年龄",
                "skills": ["技能列表"],
            }
        )
    """

    name = "document_parser"
    description = "从非结构化文本中提取结构化信息"
    system_prompt = (
        "你是一个专业的文档解析专家。"
        "请从用户提供的文本中提取结构化信息，严格按照指定的 JSON 格式返回。"
        "如果某个字段信息缺失，请填写 null。"
    )

    async def run(
        self,
        text: str,
        schema: Optional[dict] = None,
        extra_instruction: str = "",
    ) -> Optional[dict]:
        """
        解析文档文本

        :param text: 待解析的文本
        :param schema: 期望提取的字段 Schema（字典，key=字段名, value=描述）
        :param extra_instruction: 额外的解析指令
        :return: 解析后的结构化数据
        """
        if not text or not text.strip():
            return None

        # 构建提示词
        if schema:
            schema_desc = "\n".join(
                f'  "{k}": {v}' for k, v in schema.items()
            )
            prompt = (
                f"请从以下文本中提取信息，按此格式返回 JSON：\n"
                f"{{\n{schema_desc}\n}}\n\n"
                f"{extra_instruction}\n\n"
                f"文本内容：\n{text}"
            )
        else:
            prompt = (
                f"请从以下文本中提取所有关键信息，返回 JSON 格式。\n\n"
                f"{extra_instruction}\n\n"
                f"文本内容：\n{text}"
            )

        response = await self.call_llm(
            user_message=prompt,
            temperature=0.3,  # 信息提取用低温度
            max_tokens=4096,
        )

        if response.is_error:
            logger.error(f"文档解析 LLM 调用失败: {response.content}")
            return None

        # 使用 JSON 容错解析
        return JsonExtractorSkill.extract_json(response.content)
