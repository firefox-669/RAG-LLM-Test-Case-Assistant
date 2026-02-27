"""
测试用例优化器
"""

from typing import List, Dict
from config import Config
from src.rag_chain import get_rag_chain


class TestCaseOptimizer:
    """测试用例优化器类"""

    def __init__(self):
        """初始化测试用例优化器"""
        self.rag_chain = get_rag_chain()
        print("✓ 测试用例优化器初始化成功")

    def analyze_test_cases(self, test_cases: List[Dict]) -> Dict:
        """
        分析测试用例集

        Args:
            test_cases: 测试用例列表

        Returns:
            分析结果
        """
        analysis = {
            "total_count": len(test_cases),
            "by_type": {},
            "by_priority": {},
            "duplicates": [],
            "missing_fields": [],
            "suggestions": []
        }

        # 统计测试类型
        for tc in test_cases:
            test_type = tc.get('test_type', '未分类')
            analysis['by_type'][test_type] = analysis['by_type'].get(test_type, 0) + 1

            priority = tc.get('priority', '未设置')
            analysis['by_priority'][priority] = analysis['by_priority'].get(priority, 0) + 1

            # 检查缺失字段
            if not tc.get('test_name'):
                analysis['missing_fields'].append({
                    'test_id': tc.get('test_id', '未知'),
                    'field': 'test_name'
                })

        # 检测重复用例
        analysis['duplicates'] = self._detect_duplicates(test_cases)

        # 生成优化建议
        analysis['suggestions'] = self._generate_suggestions(analysis, test_cases)

        return analysis

    def optimize_test_cases(self, test_cases_text: str) -> str:
        """
        优化测试用例

        Args:
            test_cases_text: 测试用例文本

        Returns:
            优化建议
        """
        result = self.rag_chain.run(
            test_cases_text,
            Config.TESTCASE_OPTIMIZATION_PROMPT,
            top_k=3
        )

        return result['generated_text']

    def suggest_additional_cases(self, existing_cases: List[Dict], requirement: str) -> List[str]:
        """
        建议补充的测试用例

        Args:
            existing_cases: 现有测试用例
            requirement: 需求描述

        Returns:
            建议的测试场景列表
        """
        # 分析现有用例覆盖的场景
        covered_types = set(tc.get('test_type', '') for tc in existing_cases)

        suggestions = []

        # 标准测试类型
        standard_types = ['功能测试', '边界测试', '异常测试', '性能测试', '安全测试', '兼容性测试']

        for test_type in standard_types:
            if test_type not in covered_types:
                suggestions.append(f"建议添加{test_type}场景")

        # 检查常见测试场景
        common_scenarios = [
            '空值/null测试',
            '并发操作测试',
            '权限验证测试',
            '数据完整性测试',
            '错误恢复测试'
        ]

        existing_names = [tc.get('test_name', '').lower() for tc in existing_cases]

        for scenario in common_scenarios:
            if not any(scenario.lower() in name for name in existing_names):
                suggestions.append(f"建议添加：{scenario}")

        return suggestions

    def _detect_duplicates(self, test_cases: List[Dict]) -> List[Dict]:
        """检测重复的测试用例"""
        duplicates = []

        for i, tc1 in enumerate(test_cases):
            for j, tc2 in enumerate(test_cases[i+1:], i+1):
                # 简单的相似度检测
                if self._is_similar(tc1, tc2):
                    duplicates.append({
                        'case1': tc1.get('test_id', f'TC-{i+1}'),
                        'case2': tc2.get('test_id', f'TC-{j+1}'),
                        'reason': '测试步骤相似'
                    })

        return duplicates

    def _is_similar(self, tc1: Dict, tc2: Dict, threshold: float = 0.8) -> bool:
        """判断两个测试用例是否相似"""
        # 简化的相似度计算
        name1 = tc1.get('test_name', '').lower()
        name2 = tc2.get('test_name', '').lower()

        if name1 == name2:
            return True

        # 检查关键词重叠
        words1 = set(name1.split())
        words2 = set(name2.split())

        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap > threshold

        return False

    def _generate_suggestions(self, analysis: Dict, test_cases: List[Dict]) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # 检查测试类型分布
        if len(analysis['by_type']) < 3:
            suggestions.append("建议增加测试类型的多样性，当前类型较少")

        # 检查优先级分布
        high_priority = analysis['by_priority'].get('高', 0)
        if high_priority == 0:
            suggestions.append("建议标识高优先级测试用例，确保核心功能优先测试")

        # 检查重复用例
        if analysis['duplicates']:
            suggestions.append(f"发现 {len(analysis['duplicates'])} 组可能重复的测试用例，建议合并或明确差异")

        # 检查缺失字段
        if analysis['missing_fields']:
            suggestions.append(f"发现 {len(analysis['missing_fields'])} 个测试用例缺少必要字段，请补充完整")

        return suggestions


# 全局实例
_test_case_optimizer = None


def get_test_case_optimizer() -> TestCaseOptimizer:
    """获取全局测试用例优化器实例"""
    global _test_case_optimizer
    if _test_case_optimizer is None:
        _test_case_optimizer = TestCaseOptimizer()
    return _test_case_optimizer
