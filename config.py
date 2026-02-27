"""
配置管理模块
"""

import os
from pathlib import Path

# 尝试加载环境变量（如果安装了python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 未安装python-dotenv，使用默认配置")

class Config:
    """应用配置类"""

    # 应用基础配置
    APP_TITLE = os.getenv("APP_TITLE", "RAG智能测试用例助手")
    APP_PORT = int(os.getenv("APP_PORT", 8501))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    # LLM配置
    USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/chatglm3-6b")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    # 嵌入模型配置
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ChromaDB配置
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_cases")

    # RAG配置
    TOP_K = int(os.getenv("TOP_K", 5))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # 路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    TEST_CASES_DIR = DATA_DIR / "test_cases"

    # 提示词模板
    SYSTEM_PROMPT = """你是一个专业的测试工程师助手，擅长设计和优化测试用例。
你的任务是根据需求描述生成高质量的测试用例，包括功能测试、边界测试、异常测试等。

请使用以下格式输出测试用例：
测试用例ID: TC-XXX
测试用例名称: [简短描述]
测试类型: [功能/边界/异常/性能等]
优先级: [高/中/低]
前置条件: [列出前置条件]
测试步骤:
1. [步骤1]
2. [步骤2]
...
预期结果: [描述预期结果]
"""

    TESTCASE_GENERATION_PROMPT = """基于以下需求描述，生成详细的测试用例：

需求描述：
{requirement}

参考的相似测试用例：
{context}

请生成至少5个测试用例，包括：
1. 正常场景测试（Happy Path）
2. 边界值测试
3. 异常输入测试
4. 性能测试（如适用）
5. 安全性测试（如适用）
"""

    TESTCASE_OPTIMIZATION_PROMPT = """请分析以下测试用例，提供优化建议：

现有测试用例：
{test_cases}

参考的优秀案例：
{context}

请从以下方面进行分析：
1. 测试覆盖率是否完整
2. 是否存在冗余或重复的用例
3. 测试步骤是否清晰
4. 预期结果是否明确
5. 是否缺少关键场景

提供具体的优化建议和改进后的测试用例。
"""

    TESTCASE_QUERY_PROMPT = """基于以下查询问题，从知识库中找到相关的测试用例和建议：

查询问题：
{query}

相关测试用例：
{context}

请提供：
1. 与查询相关的测试用例
2. 测试建议
3. 需要注意的问题
"""

    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
        cls.TEST_CASES_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# 初始化目录
Config.ensure_dirs()
