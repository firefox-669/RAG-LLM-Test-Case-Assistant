# RAG智能测试用例助手

## 🎯 项目简介

基于RAG (Retrieval-Augmented Generation) 和大语言模型的智能测试用例助手，帮助测试工程师快速生成、优化和管理测试用例。

## ✨ 核心功能

### 1. 📤 知识库管理
- 上传产品需求文档、API文档
- 支持PDF、Word、Markdown、TXT格式
- 自动解析并建立向量索引

### 2. ✨ 智能测试用例生成
- 根据需求描述自动生成测试用例
- 支持多种测试类型（功能、边界、异常、性能、安全）
- 基于历史用例的智能推荐

### 3. 🔍 文档智能问答
- 回答关于功能逻辑、接口参数的问题
- 语义搜索相关文档
- 提供测试建议

### 4. 📊 测试用例优化
- 检测重复用例
- 评估测试覆盖率
- 提供优化建议

## 🛠️ 技术栈

- **Python 3.8+**
- **LangChain** - LLM应用开发框架
- **Sentence Transformers** - 文本嵌入模型  
- **ChromaDB** - 向量数据库
- **Streamlit** - Web UI框架

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：首次运行会自动下载嵌入模型（约500MB），请耐心等待。

### 2. 配置环境（可选）

如果要使用OpenAI API，创建`.env`文件：

```bash
cp .env.example .env
```

编辑`.env`文件，填入API Key：
```
OPENAI_API_KEY=your_api_key_here
```

**不配置也可以使用**：系统会使用内置的演示模式。

### 3. 初始化演示数据

```bash
python demo_data.py
```

这会创建包含5个测试用例和3个文档的演示知识库。

**首次运行说明**：
- 会自动下载sentence-transformers模型
- 下载时间取决于网络速度（约3-5分钟）
- 后续运行会直接使用已下载的模型

### 4. 启动应用

```bash
# 方式1：使用启动脚本
python run.py

# 方式2：直接启动Streamlit
streamlit run app.py
```

访问：http://localhost:8501

## 📁 项目结构

```
20-RAG智能测试用例助手/
├── config.py              # 配置管理
├── app.py                 # Streamlit主应用
├── demo_data.py          # 演示数据生成
├── run.py                # 启动脚本
├── requirements.txt      # 依赖包
├── README.md             # 本文件
│
├── src/                  # 核心模块
│   ├── embeddings.py     # 文本嵌入模型
│   ├── vector_store.py   # 向量数据库管理
│   ├── llm_handler.py    # LLM处理器
│   ├── rag_chain.py      # RAG链
│   ├── test_case_generator.py      # 测试用例生成器
│   ├── test_case_optimizer.py      # 用例优化器
│   └── utils.py          # 工具函数
│
└── data/                 # 数据目录
    ├── knowledge_base/   # 上传的文档
    ├── test_cases/       # 测试用例存储
    └── chroma_db/        # 向量数据库
```

## 💡 使用说明

### 步骤1：构建知识库
1. 进入"📤 知识库管理"页面
2. 上传产品需求文档、API文档
3. 系统自动解析并建立索引

### 步骤2：生成测试用例
1. 进入"✨ 测试用例生成"页面
2. 输入需求描述
3. 选择测试类型和选项
4. 点击"生成测试用例"
5. 查看、编辑和导出用例

### 步骤3：智能问答
1. 进入"🔍 智能问答"页面
2. 输入问题（如"如何测试登录功能？"）
3. 查看AI回答和参考文档

### 步骤4：优化分析
1. 进入"📊 用例优化分析"页面  
2. 粘贴现有测试用例
3. 查看分析报告和优化建议

## ❓ 常见问题

### Q1: 首次运行很慢？
**A**: 首次运行需要下载嵌入模型（约500MB），请耐心等待。后续运行会快很多。

### Q2: 如何使用自己的文档？
**A**: 在"知识库管理"页面上传您的PDF、Word或文本文档即可。

### Q3: 需要OpenAI API吗？
**A**: 不是必需的。不配置API也可以使用，系统会使用演示模式生成测试用例模板。

### Q4: 生成的用例质量如何提高？
**A**: 
- 上传更多高质量的参考文档
- 提供详细的需求描述
- 配置OpenAI API使用GPT模型

### Q5: 数据存储在哪里？
**A**: 所有数据存储在`data/`目录下，完全本地化，不会上传到外部服务器。

## 🔧 故障排除

### 问题1: 模块导入错误
```bash
# 确保在项目根目录运行
cd "D:\机器学习实训营\20-RAG智能测试用例助手"
pip install -r requirements.txt
```

### 问题2: ChromaDB错误
```bash
# 删除数据库重新初始化
rm -rf data/chroma_db
python demo_data.py
```

### 问题3: 模型下载失败
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers
```

## 📚 更多文档

- `QUICKSTART.md` - 详细的快速开始指南
- `PROJECT_REPORT.md` - 完整的项目技术报告
- `COMPLETE_SUMMARY.md` - 项目完成总结

## 🎓 学习资源

- [LangChain文档](https://python.langchain.com/)
- [Sentence Transformers文档](https://www.sbert.net/)
- [ChromaDB文档](https://docs.trychroma.com/)
- [Streamlit文档](https://docs.streamlit.io/)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**祝使用愉快！** 🎉

如有问题，请查看文档或提交Issue。
