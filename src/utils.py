"""
工具函数模块
"""

import re
from typing import List, Dict
import pandas as pd
from pathlib import Path


def format_test_case_as_text(test_case: Dict) -> str:
    """
    将测试用例格式化为文本

    Args:
        test_case: 测试用例字典

    Returns:
        格式化的文本
    """
    text = f"""
测试用例ID: {test_case.get('test_id', 'N/A')}
测试用例名称: {test_case.get('test_name', 'N/A')}
测试类型: {test_case.get('test_type', 'N/A')}
优先级: {test_case.get('priority', 'N/A')}

前置条件:
{test_case.get('preconditions', 'N/A')}

测试步骤:
{test_case.get('steps', 'N/A')}

预期结果:
{test_case.get('expected_result', 'N/A')}
"""
    return text.strip()


def test_cases_to_dataframe(test_cases: List[Dict]) -> pd.DataFrame:
    """
    将测试用例列表转换为DataFrame

    Args:
        test_cases: 测试用例列表

    Returns:
        DataFrame
    """
    if not test_cases:
        return pd.DataFrame()

    return pd.DataFrame(test_cases)


def export_test_cases_to_csv(test_cases: List[Dict], filepath: str):
    """
    导出测试用例到CSV

    Args:
        test_cases: 测试用例列表
        filepath: 文件路径
    """
    df = test_cases_to_dataframe(test_cases)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"✓ 测试用例已导出到: {filepath}")


def export_test_cases_to_excel(test_cases: List[Dict], filepath: str):
    """
    导出测试用例到Excel

    Args:
        test_cases: 测试用例列表
        filepath: 文件路径
    """
    df = test_cases_to_dataframe(test_cases)
    df.to_excel(filepath, index=False, engine='openpyxl')
    print(f"✓ 测试用例已导出到: {filepath}")


def parse_document_content(file_path: str) -> str:
    """
    解析文档内容

    Args:
        file_path: 文件路径

    Returns:
        文档内容
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif suffix == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif suffix == '.pdf':
            return parse_pdf(file_path)
        elif suffix in ['.doc', '.docx']:
            return parse_word(file_path)
        else:
            return f"不支持的文件格式: {suffix}"
    except Exception as e:
        return f"解析文件失败: {str(e)}"


def parse_pdf(file_path: str) -> str:
    """解析PDF文件"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)
    except ImportError:
        return "需要安装 pypdf: pip install pypdf"
    except Exception as e:
        return f"解析PDF失败: {str(e)}"


def parse_word(file_path: str) -> str:
    """解析Word文档"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except ImportError:
        return "需要安装 python-docx: pip install python-docx"
    except Exception as e:
        return f"解析Word失败: {str(e)}"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本分块

    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap: 重叠大小

    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界分割
        if end < text_len:
            last_period = chunk.rfind('。')
            last_newline = chunk.rfind('\n')
            split_point = max(last_period, last_newline)

            if split_point > chunk_size * 0.5:  # 至少保留一半
                chunk = chunk[:split_point + 1]
                end = start + split_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def extract_api_info(text: str) -> List[Dict]:
    """
    从文档中提取API信息

    Args:
        text: 文档文本

    Returns:
        API信息列表
    """
    apis = []

    # 简单的API提取规则（可以根据实际格式调整）
    api_pattern = r'(?:接口|API)[：:]\s*(.+?)(?:\n|$)'
    matches = re.finditer(api_pattern, text)

    for match in matches:
        api_name = match.group(1).strip()
        apis.append({
            'name': api_name,
            'context': text[max(0, match.start()-200):match.end()+200]
        })

    return apis


def calculate_coverage(test_cases: List[Dict], requirements: List[str]) -> Dict:
    """
    计算测试覆盖率

    Args:
        test_cases: 测试用例列表
        requirements: 需求列表

    Returns:
        覆盖率统计
    """
    covered = set()

    for tc in test_cases:
        requirement = tc.get('requirement', '')
        if requirement:
            covered.add(requirement)

    coverage = {
        'total_requirements': len(requirements),
        'covered_requirements': len(covered),
        'coverage_rate': len(covered) / len(requirements) if requirements else 0,
        'uncovered': [r for r in requirements if r not in covered]
    }

    return coverage


def validate_test_case(test_case: Dict) -> List[str]:
    """
    验证测试用例的完整性

    Args:
        test_case: 测试用例字典

    Returns:
        错误信息列表
    """
    errors = []

    required_fields = ['test_id', 'test_name', 'test_type', 'steps', 'expected_result']

    for field in required_fields:
        if not test_case.get(field):
            errors.append(f"缺少必填字段: {field}")

    # 验证测试步骤
    steps = test_case.get('steps', '')
    if steps and not any(char.isdigit() for char in steps):
        errors.append("测试步骤应包含编号")

    return errors
