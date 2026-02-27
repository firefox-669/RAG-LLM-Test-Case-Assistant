# -*- coding: utf-8 -*-
"""
Test Case Generator - Simplified
"""

from typing import List, Dict, Optional
from config import Config
from src.rag_chain import get_rag_chain
import re


class TestCaseGenerator:
    def __init__(self):
        self.rag_chain = get_rag_chain()
        print("[OK] Test case generator initialized")

    def generate_from_requirement(
        self,
        requirement: str,
        test_types: Optional[List[str]] = None,
        priority: str = "Medium",
        num_cases: int = 5
    ) -> List[Dict]:
        result = self.rag_chain.run(requirement, Config.TESTCASE_GENERATION_PROMPT, top_k=3)
        test_cases = self._parse_generated_cases(result['generated_text'], requirement)

        for tc in test_cases:
            if 'priority' not in tc or not tc['priority']:
                tc['priority'] = priority

        return test_cases

    def generate_boundary_tests(self, requirement: str) -> List[Dict]:
        boundary_prompt = f"Generate boundary test cases for: {requirement}"
        return self.generate_from_requirement(boundary_prompt, test_types=["Boundary Test"])

    def generate_exception_tests(self, requirement: str) -> List[Dict]:
        exception_prompt = f"Generate exception test cases for: {requirement}"
        return self.generate_from_requirement(exception_prompt, test_types=["Exception Test"])

    def _parse_generated_cases(self, generated_text: str, requirement: str) -> List[Dict]:
        test_cases = []
        cases_text = re.split(r'\n---+\n|\n\nTest Case', generated_text)

        for i, case_text in enumerate(cases_text):
            if not case_text.strip():
                continue

            test_case = self._parse_single_case(case_text, i + 1)
            if test_case:
                test_case['requirement'] = requirement
                test_cases.append(test_case)

        return test_cases

    def _parse_single_case(self, case_text: str, index: int) -> Optional[Dict]:
        try:
            test_case = {
                'test_id': '',
                'test_name': '',
                'test_type': '',
                'priority': '',
                'preconditions': '',
                'steps': '',
                'expected_result': ''
            }

            id_match = re.search(r'(?:Test Case ID|测试用例ID)[：:]\s*(.+?)(?:\n|$)', case_text)
            if id_match:
                test_case['test_id'] = id_match.group(1).strip()
            else:
                test_case['test_id'] = f'TC-{index:03d}'

            name_match = re.search(r'(?:Test Case Name|测试用例名称)[：:]\s*(.+?)(?:\n|$)', case_text)
            if name_match:
                test_case['test_name'] = name_match.group(1).strip()

            type_match = re.search(r'(?:Test Type|测试类型)[：:]\s*(.+?)(?:\n|$)', case_text)
            if type_match:
                test_case['test_type'] = type_match.group(1).strip()

            priority_match = re.search(r'(?:Priority|优先级)[：:]\s*(.+?)(?:\n|$)', case_text)
            if priority_match:
                test_case['priority'] = priority_match.group(1).strip()

            precond_match = re.search(r'(?:Preconditions|前置条件)[：:](.*?)(?=(?:Test Steps|测试步骤|Expected Result|预期结果)|$)', case_text, re.DOTALL)
            if precond_match:
                test_case['preconditions'] = precond_match.group(1).strip()

            steps_match = re.search(r'(?:Test Steps|测试步骤)[：:](.*?)(?=(?:Expected Result|预期结果)|$)', case_text, re.DOTALL)
            if steps_match:
                test_case['steps'] = steps_match.group(1).strip()

            result_match = re.search(r'(?:Expected Result|预期结果)[：:](.*?)$', case_text, re.DOTALL)
            if result_match:
                test_case['expected_result'] = result_match.group(1).strip()

            return test_case if test_case['test_name'] else None

        except Exception as e:
            print(f"Parse test case failed: {e}")
            return None


_test_case_generator = None

def get_test_case_generator() -> TestCaseGenerator:
    global _test_case_generator
    if _test_case_generator is None:
        _test_case_generator = TestCaseGenerator()
    return _test_case_generator
