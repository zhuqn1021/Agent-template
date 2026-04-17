"""
报告生成技能

利用 LLM 基于模板生成结构化报告。
"""
import logging
from typing import Optional

from skills.base import Skill

logger = logging.getLogger(__name__)


class ReportGeneratorSkill(Skill):
    """
    报告生成技能

    支持基于提示词模板生成结构化报告，模板中可使用 {variable} 占位符。

    使用示例：
        skill = ReportGeneratorSkill(llm)

        # 使用自定义模板
        report = await skill.run(
            template="请为 {name} 生成一份面试评估报告...",
            variables={"name": "张三", "position": "Python 开发"},
        )

        # 直接传入提示词
        report = await skill.run(prompt="请总结以下会议纪要...")
    """

    name = "report_generator"
    description = "基于模板生成结构化报告"
    system_prompt = (
        "你是一个专业的报告撰写专家。"
        "请根据用户提供的信息，生成一份结构化、专业、客观的报告。"
        "报告语言要简洁专业，适合商务场景阅读。"
    )

    async def run(
        self,
        template: Optional[str] = None,
        variables: Optional[dict] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        生成报告

        :param template: 报告模板（含 {variable} 占位符）
        :param variables: 模板变量字典
        :param prompt: 直接提示词（与 template 二选一）
        :param system_prompt: 自定义系统提示词
        :param max_tokens: 最大 token 数
        :return: 生成的报告文本
        """
        # 构建最终 prompt
        if template and variables:
            try:
                final_prompt = template.format(**variables)
            except KeyError as e:
                logger.error(f"模板变量缺失: {e}")
                return f"[错误] 模板变量缺失: {e}"
        elif prompt:
            final_prompt = prompt
        else:
            return "[错误] 必须提供 template+variables 或 prompt"

        response = await self.call_llm(
            user_message=final_prompt,
            system_message=system_prompt,
            temperature=0.7,
            max_tokens=max_tokens,
        )

        if response.is_error:
            logger.error(f"报告生成失败: {response.content}")
            return f"[错误] {response.content}"

        return response.content
