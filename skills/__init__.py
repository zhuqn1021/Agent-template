"""
Skills 技能系统

可组合的能力单元，封装常见 AI 任务模式。
"""

from skills.base import Skill
from skills.json_extractor import JsonExtractorSkill
from skills.document_parser import DocumentParserSkill
from skills.report_generator import ReportGeneratorSkill

__all__ = ["Skill", "JsonExtractorSkill", "DocumentParserSkill", "ReportGeneratorSkill"]
