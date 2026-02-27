# 📖 使用指南

## 🚀 快速开始

### 前置要求
- Python 3.8+
- pip 包管理器
- 网络连接（用于下载模型）

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 运行应用
```bash
streamlit run app_cn.py
```

### 3️⃣ 打开浏览器
访问 `http://localhost:8501`

---

## 💡 功能使用

### 🏠 首页
- 显示项目概览和核心特性
- 包含搜索准确率、文件格式支持等统计信息

### 📤 上传文档
**步骤**：
1. 点击"选择文档"按钮
2. 选择 TXT、MD、PDF 或 DOCX 格式的文件
3. 查看文件内容预览
4. 点击"✅ 添加到知识库"

**支持的格式**：
- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.pdf` - PDF 文档（需要 pdfplumber）
- `.docx` - Word 文档（需要 python-docx）

### 🔍 智能问答
**步骤**：
1. 进入"智能问答"页面
2. 输入你的问题
3. 选择响应模式：
   - **纯检索**：直接返回相关文档
   - **LLM智能问答**：使用大语言模型生成答案
4. 点击"🔍 搜索"

**例子**：
```
问：登录功能的参数有哪些？
答：系统会搜索知识库，返回相关的登录参数说明
```

### ✨ 生成测试用例
**步骤**：
1. 进入"生成用例"页面
2. 输入或粘贴产品需求文档
3. 点击"🚀 生成用例"
4. 自动生成测试用例

**生成的用例包括**：
- 正常情况测试
- 异常情况处理
- 边界值测试
- 优先级标记

### ℹ️ 系统信息
显示当前系统配置和运行状态：
- 嵌入维度
- 文本分割大小
- 向量库路径
- 文档数量统计

---

## 🔧 配置调整

编辑 `config.py` 文件调整参数：

```python
# 向量存储配置
CHROMA_DB_PATH = "data/knowledge_base/chroma_db"

# 文本处理配置
CHUNK_SIZE = 1000          # 分割大小（字符）
CHUNK_OVERLAP = 200        # 分割重叠（字符）

# 搜索配置
SEARCH_TOP_K = 3           # 返回结果数量
VECTOR_WEIGHT = 0.5        # 向量权重
KEYWORD_WEIGHT = 0.5       # 关键词权重

# LLM配置
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-3.5-turbo"
```

---

## 📊 数据管理

### 文件位置
```
data/
├── chroma_db/              # 向量数据库
├── knowledge_base/         # 上传的文档
└── test_cases/             # 生成的测试用例
```

### 清空数据
```bash
# 清空向量库
rm -r data/chroma_db/*

# 清空上传的文档
rm -r data/knowledge_base/*

# 清空生成的用例
rm -r data/test_cases/*
```

---

## 🔍 搜索优化建议

### 1. 查询语句
✅ **好的查询**：
- "用户登录的参数说明"
- "API 返回值格式"
- "错误处理机制"

❌ **不好的查询**：
- "什么"
- "嗯"
- "哈哈"

### 2. 文档准备
✅ **好的文档**：
- 结构清晰，有标题
- 包含完整的功能说明
- 有具体的参数定义

❌ **不好的文档**：
- 纯图片
- 没有结构的文本堆砌
- 过度简化的说明

### 3. 混合搜索策略
系统同时使用：
- **向量相似度**：理解查询的语义
- **关键词匹配**：确保精确性
- **加权综合**：各占50%

---

## 🎓 高级用法

### 自定义嵌入模型
编辑 `src/embeddings.py`：
```python
from sentence_transformers import SentenceTransformer

class CustomEmbedding:
    def __init__(self):
        # 更换模型
        self.model = SentenceTransformer('your-model-name')
    
    def embed(self, text):
        return self.model.encode(text)
```

### 集成自己的LLM
编辑 `src/rag_chain.py`：
```python
def run_with_custom_llm(self, query, llm_func):
    # 检索相关文档
    docs = self.retrieve(query)
    
    # 调用自定义LLM
    return llm_func(query, docs)
```

---

## ⚠️ 常见问题

### Q1: 模型下载失败
**问题**：网络超时，无法下载 Sentence Transformers 模型
**解决**：
```bash
# 设置离线模式
export HF_HUB_OFFLINE=1

# 或使用本地模型
# 在 config.py 中修改模型路径
```

### Q2: 内存不足
**问题**：加载大文档时内存溢出
**解决**：
```python
# 减小 CHUNK_SIZE
CHUNK_SIZE = 500  # 改小

# 减小返回结果数
SEARCH_TOP_K = 2
```

### Q3: 搜索结果不相关
**问题**：搜索返回的文档与查询不匹配
**解决**：
1. 调整权重比例
2. 优化文档分割
3. 检查文档质量

### Q4: 应用启动慢
**问题**：Streamlit 启动需要很长时间
**解决**：
```bash
# 使用缓存加速
streamlit run app_cn.py --client.showErrorDetails false
```

---

## 🚀 部署到生产环境

### Streamlit Cloud 部署
1. 上传到 GitHub
2. 访问 streamlit.io/cloud
3. 连接 GitHub 仓库
4. 自动部署

### 本地服务器部署
```bash
# 使用 Gunicorn
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:8501 "streamlit run app_cn.py"
```

### Docker 部署
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app_cn.py"]
```

---

## 📚 扩展功能

### 添加数据库支持
在 `src/vector_store.py` 中扩展

### 集成更多LLM
在 `src/llm_handler.py` 中添加

### 自定义搜索算法
在 `src/rag_chain.py` 中修改

---

## 📞 获取帮助

- **查看日志**：`streamlit_log.txt`
- **检查配置**：`config.py`
- **查看文档**：`README.md`、`PROJECT_INFO.md`

---

**最后更新**：2026年2月27日
**版本**：1.0
**状态**：✅ 正式版

