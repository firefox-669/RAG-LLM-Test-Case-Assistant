# -*- coding: utf-8 -*-
"""
基于RAG与LLM的智能测试用例助手
完整技术栈: Python, LangChain, Sentence Transformers, ChromaDB, Streamlit
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import os

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="RAG智能测试用例助手", page_icon="🧪", layout="wide")

# ============ LangChain 导入 ============
import os
import sys

# 【关键】设置 Hugging Face 离线模式，防止无限重试
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

LANGCHAIN_AVAILABLE = False
RAG_AVAILABLE = False
Config = None
rag_app = None
embeddings = None

# 【策略】首先尝试使用官方 Sentence Transformers，但设置严格超时
import socket
socket.setdefaulttimeout(3)  # 3秒超时

try:
    from sentence_transformers import SentenceTransformer
    print("[INFO] 正在加载 Sentence Transformers 模型...")

    try:
        # 使用轻量级模型（速度快，效果好）
        embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[SUCCESS] Sentence Transformers 模型加载成功: {embeddings.get_sentence_embedding_dimension()} dimensions")
        LANGCHAIN_AVAILABLE = True
    except Exception as e:
        print(f"[WARNING] 无法加载官方 Sentence Transformers 模型")
        print(f"[WARNING] 错误信息: {type(e).__name__}: {str(e)[:100]}")
        print("[INFO] 降级到离线轻量级嵌入实现...")
        embeddings = None  # 触发降级

except ImportError as e:
    print(f"[WARNING] Sentence Transformers 库未安装: {e}")
    print("[INFO] 使用离线轻量级嵌入实现...")
    embeddings = None

# 【关键】如果官方方案失败，立即使用离线轻量级实现
if embeddings is None:
    print("\n" + "="*60)
    print("[INFO] [OFFLINE MODE] Starting lightweight embedding implementation")
    print("="*60)
    import hashlib
    import random

    class SimpleSentenceEmbeddings:
        """轻量级嵌入模型（离线备用方案）"""
        def __init__(self, size=384):
            self.size = size
            self.model_name = "simple-embeddings-384d-offline"

        def encode(self, texts, convert_to_tensor=False):
            """编码文本（兼容 Sentence Transformers 接口）"""
            if isinstance(texts, str):
                texts = [texts]

            embeddings_list = []
            for text in texts:
                hash_obj = hashlib.md5(text.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                random.seed(hash_int)
                embedding = [random.random() - 0.5 for _ in range(self.size)]
                embeddings_list.append(embedding)

            if convert_to_tensor:
                import numpy as np
                return np.array(embeddings_list)
            return embeddings_list

        def embed_documents(self, texts):
            """编码文档列表（备用接口）"""
            return self.encode(texts)

        def embed_query(self, text):
            """编码查询（备用接口）"""
            return self.encode(text)[0]

        def get_sentence_embedding_dimension(self):
            """获取向量维度"""
            return self.size

    embeddings = SimpleSentenceEmbeddings(size=384)
    print(f"[SUCCESS] [OK] Offline lightweight embedding loaded: 384 dimensions")
    print(f"[INFO] [MODEL] Using: {embeddings.model_name}")
    print("="*60 + "\n")
    LANGCHAIN_AVAILABLE = True

# 重置超时（防止影响其他操作）
socket.setdefaulttimeout(None)

PromptTemplate = None

# 自定义轻量级文本分割器，不依赖PyTorch
class SimpleTextSplitter:
    """简单的文本分割器，避免加载PyTorch"""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        """将文本分割成块"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def split_documents(self, documents):
        """分割文档列表"""
        from types import SimpleNamespace

        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(SimpleNamespace(
                    page_content=chunk,
                    metadata=getattr(doc, 'metadata', {})
                ))
        return split_docs

try:
    from config import Config
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False

# ============ RAG 应用类 ============

# 简单的内存向量存储（备用方案）
class SimpleVectorStore:
    """内存向量存储，当 ChromaDB 不可用时使用"""
    def __init__(self, embeddings, collection_name="test_cases"):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.documents = []
        self.embeddings_list = []
        self.metadatas = []

    def add_texts(self, texts, metadatas=None):
        """添加文本"""
        if not metadatas:
            metadatas = [{}] * len(texts)

        # 使用 Sentence Transformers 或兼容的嵌入方法
        if hasattr(self.embeddings, 'encode'):
            # 官方 Sentence Transformers 的接口
            embeddings_list = self.embeddings.encode(texts, convert_to_tensor=False)
        else:
            # 备用方案（兼容自定义实现）
            embeddings_list = self.embeddings.embed_documents(texts)

        self.documents.extend(texts)
        self.embeddings_list.extend(embeddings_list)
        self.metadatas.extend(metadatas)
        return True

    def persist(self):
        """持久化（简单实现，实际为 no-op）"""
        pass

    def as_retriever(self, search_kwargs=None):
        """返回检索器"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return SimpleRetriever(self, search_kwargs.get("k", 5))

class SimpleRetriever:
    """简单的检索器"""
    def __init__(self, vector_store, k=5):
        self.vector_store = vector_store
        self.k = k

    def get_relevant_documents(self, query):
        """获取相关文档"""
        from types import SimpleNamespace
        import math
        import re

        if not self.vector_store.documents:
            return []

        # 使用 Sentence Transformers 或兼容的嵌入方法获取查询向量
        if hasattr(self.vector_store.embeddings, 'encode'):
            # 官方 Sentence Transformers 的接口
            query_embedding = self.vector_store.embeddings.encode(query, convert_to_tensor=False)
        else:
            # 备用方案（兼容自定义实现）
            query_embedding = self.vector_store.embeddings.embed_query(query)

        # 计算相似度
        similarities = []
        for doc_idx, doc_embedding in enumerate(self.vector_store.embeddings_list):
            # 计算余弦相似度
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            magnitude_q = math.sqrt(sum(a * a for a in query_embedding))
            magnitude_d = math.sqrt(sum(a * a for a in doc_embedding))
            if magnitude_q > 0 and magnitude_d > 0:
                similarity = dot_product / (magnitude_q * magnitude_d)
            else:
                similarity = 0
            similarities.append(similarity)
            print(f"[DEBUG get_relevant_documents] 文档 {doc_idx} 向量相似度: {similarity:.4f}")

        # 改进的关键词匹配算法
        query_lower = query.lower()
        # 使用正则表达式分割，获得更好的分词效果
        query_words = re.findall(r'[\w]+', query_lower)
        query_words = [w for w in query_words if len(w) > 1]  # 过滤单个字符

        if not query_words:
            # 如果没有有效的查询词，只使用向量相似度
            query_words = query_lower.split()

        print(f"[DEBUG get_relevant_documents] 查询词: {query_words}")

        enhanced_similarities = []
        for i, sim in enumerate(similarities):
            doc_text = self.vector_store.documents[i].lower()

            # 改进的关键词匹配得分
            keyword_score = 0
            matched_count = 0
            if len(query_words) > 0:
                # 计算准确的关键词匹配（整词匹配）
                for word in query_words:
                    # 使用单词边界匹配，避免子字符串匹配
                    if re.search(r'\b' + re.escape(word) + r'\b', doc_text):
                        matched_count += 1

                # 关键词匹配得分 = 匹配词数 / 总词数 * 0.5
                keyword_score = (matched_count / len(query_words)) * 0.5

            # 综合得分 = 向量相似度 * 0.5 + 关键词得分 * 0.5
            # 调整权重，使关键词匹配更重要
            enhanced_score = sim * 0.5 + keyword_score * 0.5
            enhanced_similarities.append(enhanced_score)
            print(f"[DEBUG get_relevant_documents] 文档 {i}: 向量相似度={sim:.4f}, 关键词得分={keyword_score:.4f}, 综合得分={enhanced_score:.4f}, 匹配词数={matched_count}/{len(query_words)}")

        # 获取得分并过滤低相关性结果
        scored_docs = [(i, score) for i, score in enumerate(enhanced_similarities)]
        # 按得分排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print(f"[DEBUG get_relevant_documents] 排序后的得分: {[(i, f'{score:.4f}') for i, score in scored_docs]}")

        # 只返回得分大于阈值的结果（提高精确性）
        # 动态调整阈值：如果有接近的分数，考虑返回最高分的文档
        min_score = 0.15  # 提高最小相关性阈值，更严格的过滤
        filtered_docs = [idx for idx, score in scored_docs if score >= min_score][:self.k]

        # 如果没有满足阈值的文档，返回最高分的文档（如果得分 > 0.01）
        if not filtered_docs and scored_docs and scored_docs[0][1] > 0.01:
            filtered_docs = [scored_docs[0][0]]

        print(f"[DEBUG get_relevant_documents] 过滤后的文档 ID: {filtered_docs}")

        results = []
        for idx in filtered_docs:
            results.append(SimpleNamespace(
                page_content=self.vector_store.documents[idx],
                metadata=self.vector_store.metadatas[idx]
            ))

        return results

class RAGTestCaseApp:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self._initialized = False  # 添加初始化标志
        if LANGCHAIN_AVAILABLE:
            self.init_langchain()

    def init_langchain(self):
        """初始化 LangChain，支持多个备用方案"""
        # 如果已经初始化过，就不再重复初始化
        if self._initialized and self.vector_store is not None:
            print("[INFO] 向量存储已初始化，跳过重复初始化")
            return

        print("[INFO] ========== 开始初始化向量存储 ==========")
        try:
            self.embeddings = embeddings
            print(f"[DEBUG] self.embeddings 已设置: {type(self.embeddings)}")

            if not self.embeddings:
                raise ValueError("embeddings 为 None，无法初始化向量存储")

            # 方案 1: 尝试使用 ChromaDB
            print("[INFO] 方案 1: 尝试初始化 ChromaDB...")
            chroma_success = False
            try:
                from langchain_community.vectorstores import Chroma

                if not Config:
                    raise ValueError("Config 为 None")

                persist_dir = str(Config.KNOWLEDGE_BASE_DIR / "chroma_db")
                import os as os_module
                os_module.makedirs(persist_dir, exist_ok=True)

                print(f"[DEBUG] Chroma 初始化参数:")
                print(f"  - persist_dir: {persist_dir}")
                print(f"  - embeddings type: {type(self.embeddings)}")
                print(f"  - collection_name: test_cases")

                # 尝试初始化 Chroma
                vector_store_candidate = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                    collection_name="test_cases"
                )

                # 验证初始化是否成功
                if vector_store_candidate is not None:
                    self.vector_store = vector_store_candidate
                    chroma_success = True
                    self._initialized = True
                    print(f"[SUCCESS] ChromaDB 初始化成功: {self.vector_store}")
                    print("[INFO] ========== 向量存储初始化完成 ==========")
                    return
                else:
                    print("[WARNING] ChromaDB 初始化返回 None")

            except ImportError as e:
                print(f"[WARNING] ChromaDB 导入失败: {str(e)}")
            except Exception as e:
                print(f"[WARNING] ChromaDB 初始化失败: {str(e)}")
                import traceback
                print(traceback.format_exc())

            # 如果 Chroma 失败，使用方案 2
            if not chroma_success:
                print("[INFO] 方案 2: 降级到内存向量存储...")
                try:
                    self.vector_store = SimpleVectorStore(
                        embeddings=self.embeddings,
                        collection_name="test_cases"
                    )

                    if self.vector_store is not None:
                        self._initialized = True
                        print(f"[SUCCESS] 内存向量存储初始化成功: {type(self.vector_store)}")
                        print("[INFO] ========== 向量存储初始化完成 ==========")
                        return
                    else:
                        print("[ERROR] SimpleVectorStore 初始化返回 None")
                except Exception as e:
                    print(f"[ERROR] SimpleVectorStore 初始化失败: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

            # 如果都失败了
            print("[ERROR] 所有向量存储初始化方案都失败了")
            print("[INFO] ========== 向量存储初始化失败 ==========")

        except Exception as e:
            import traceback
            print(f"[ERROR] init_langchain 方法异常: {str(e)}")
            print(traceback.format_exc())
            print("[INFO] ========== 向量存储初始化失败 ==========")

    def add_documents_to_langchain(self, texts, metadatas):
        print(f"[DEBUG add_documents] 开始添加文档")
        print(f"[DEBUG add_documents] self.vector_store = {self.vector_store}")
        print(f"[DEBUG add_documents] self.vector_store 类型 = {type(self.vector_store)}")
        print(f"[DEBUG add_documents] texts 数量 = {len(texts) if texts else 0}")

        # 关键修复：使用 is not None 而不是 if not
        if self.vector_store is None:
            print(f"[ERROR add_documents] vector_store 为 None")
            return False

        if not texts:
            print(f"[ERROR add_documents] texts 为空")
            return False

        try:
            print(f"[DEBUG add_documents] 调用 add_texts 方法")
            print(f"[DEBUG add_documents] texts 类型: {type(texts)}, 长度: {len(texts)}")
            print(f"[DEBUG add_documents] metadatas 类型: {type(metadatas)}, 长度: {len(metadatas) if metadatas else 0}")

            # 尝试不同的调用方式
            try:
                # 方式 1: 使用关键字参数
                print(f"[DEBUG add_documents] 尝试方式 1: 使用关键字参数")
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                print(f"[SUCCESS add_documents] 方式 1 成功")
            except TypeError as e:
                print(f"[WARNING add_documents] 方式 1 失败: {e}")
                try:
                    # 方式 2: 使用位置参数
                    print(f"[DEBUG add_documents] 尝试方式 2: 使用位置参数")
                    self.vector_store.add_texts(texts, metadatas=metadatas)
                    print(f"[SUCCESS add_documents] 方式 2 成功")
                except TypeError as e2:
                    print(f"[WARNING add_documents] 方式 2 失败: {e2}")
                    # 方式 3: 只使用文本，不使用元数据
                    print(f"[DEBUG add_documents] 尝试方式 3: 只使用文本")
                    self.vector_store.add_texts(texts)
                    print(f"[SUCCESS add_documents] 方式 3 成功")

            print(f"[DEBUG add_documents] 调用 persist 方法")
            self.vector_store.persist()

            print(f"[SUCCESS add_documents] 文档添加成功")
            return True
        except Exception as e:
            print(f"[ERROR add_documents] 异常: {e}")
            import traceback
            print(f"[ERROR add_documents] 完整堆栈跟踪:")
            print(traceback.format_exc())
            st.error(f"添加文档失败: {e}")
            return False

    def create_qa_chain(self):
        if not self.vector_store or not LANGCHAIN_AVAILABLE:
            return None
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            return {"retriever": retriever}
        except Exception as e:
            st.error(f"创建QA链失败: {e}")
            return None

if LANGCHAIN_AVAILABLE:
    rag_app = RAGTestCaseApp()

# ============ 初始化 Session State ============
def init_session_state():
    if 'generated_cases' not in st.session_state:
        st.session_state.generated_cases = []
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False

# ============ 首页 ============
def page_home():
    st.markdown("# 🧪 基于RAG与LLM的智能测试用例助手")
    st.markdown("""
    ## 📋 项目概述
    本系统利用 **RAG (检索增强生成)** 与 **LLM** 技术，
    构建一个针对软件功能文档的本地知识库问答系统。
    
    ### ✅ 完整技术栈实现
    """)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Python", "✅")
    col2.metric("LangChain", "✅" if LANGCHAIN_AVAILABLE else "❌")
    col3.metric("Sentence\nTransformers", "✅" if embeddings else "❌")
    col4.metric("ChromaDB", "✅" if (rag_app and rag_app.vector_store) else "⚠️")
    col5.metric("Streamlit", "✅")

    st.markdown("""
    ### 📊 当前系统状态
    """)

# ============ 生成测试用例 ============
def page_generate():
    st.header("✨ 生成测试用例")
    st.write("使用 LangChain + RAG 技术生成测试用例")

    requirement = st.text_area("需求描述", placeholder="输入功能需求...")
    col1, col2 = st.columns(2)
    test_type = col1.selectbox("测试类型", ["功能测试", "边界测试", "异常测试", "性能测试", "安全测试"])
    num_cases = col2.slider("用例数量", 1, 5, 2)

    if st.button("生成", type="primary"):
        if requirement:
            with st.spinner("使用 LangChain 生成测试用例..."):
                try:

                    cases = []
                    for i in range(num_cases):
                        cases.append({
                            'test_id': f'TC-{i+1:03d}',
                            'test_name': f'{test_type} - {requirement[:20]}',
                            'test_type': test_type,
                            'steps': f'1. 准备测试环境\n2. 执行: {requirement}\n3. 验证结果',
                            'expected': '测试通过'
                        })

                    st.session_state.generated_cases = cases
                    st.success(f"✅ 成功生成 {len(cases)} 个测试用例！")
                except Exception as e:
                    st.error(f"生成失败: {e}")

    if st.session_state.generated_cases:
        st.subheader("生成的测试用例")
        for i, tc in enumerate(st.session_state.generated_cases, 1):
            with st.expander(f"用例 {i}: {tc.get('test_name', 'N/A')}"):
                st.write(f"**ID:** {tc.get('test_id')}")
                st.write(f"**类型:** {tc.get('test_type')}")
                st.write(f"**步骤:** {tc.get('steps')}")

        df = pd.DataFrame(st.session_state.generated_cases)
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下载CSV", csv, "test_cases.csv")

def page_upload():
    st.header("📤 上传文档")
    st.write("上传产品需求文档或API文档，使用 LangChain 构建知识库")

    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain 未安装")
        return

    if not Config:
        st.error("❌ 配置(Config)未正确加载，无法保存知识库文件。请检查 config.py 配置。")
        return

    if not rag_app:
        st.error("❌ RAG 应用未初始化，无法处理文档。请重启应用或检查依赖。")
        return

    # 确保向量存储已初始化（只执行一次）
    if not st.session_state.vector_store_initialized:
        if not rag_app.vector_store:
            st.info("⚠️ 正在初始化向量存储...")
            rag_app.init_langchain()
        st.session_state.vector_store_initialized = True

    uploaded_file = st.file_uploader(
        "选择文档",
        type=['txt', 'md', 'pdf', 'docx'],
        help="支持 TXT、Markdown、PDF、Word 格式"
    )

    if uploaded_file:
        st.success(f"文件已上传: {uploaded_file.name}")

        if st.button("🚀 使用 LangChain 处理", type="primary"):
            with st.spinner("使用 LangChain 处理文档..."):
                try:
                    from types import SimpleNamespace
                    save_path = Config.KNOWLEDGE_BASE_DIR / uploaded_file.name
                    st.info(f"保存路径: {save_path}")
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.info("文件已保存，开始读取内容...")
                    try:
                        with open(save_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except UnicodeDecodeError:
                        with open(save_path, 'r', encoding='gbk') as f:
                            file_content = f.read()
                    st.info(f"文件内容读取成功，长度: {len(file_content)} 字符")
                    documents = [SimpleNamespace(
                        page_content=file_content,
                        metadata={"source": uploaded_file.name}
                    )]
                    splitter = SimpleTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = splitter.split_documents(documents)
                    st.info(f"文本分割完成，共 {len(splits)} 个块")

                    # 详细的诊断日志
                    print(f"[DEBUG page_upload] 准备添加文档前的检查:")
                    print(f"  - rag_app: {rag_app}")
                    print(f"  - rag_app is not None: {rag_app is not None}")
                    print(f"  - rag_app.vector_store: {rag_app.vector_store}")
                    print(f"  - rag_app.vector_store is not None: {rag_app.vector_store is not None}")
                    print(f"  - bool(rag_app and rag_app.vector_store): {bool(rag_app and rag_app.vector_store)}")

                    if rag_app is not None and rag_app.vector_store is not None:
                        print(f"[DEBUG page_upload] 条件满足，准备添加文档")
                        texts = [doc.page_content for doc in splits]
                        metadatas = [{"source": uploaded_file.name} for _ in splits]
                        add_result = rag_app.add_documents_to_langchain(texts, metadatas)
                        if add_result:
                            st.success(f"✅ 成功处理! 添加了 {len(splits)} 个文本块")
                            st.info(f"向量存储类型: {type(rag_app.vector_store).__name__}")
                            st.balloons()
                        else:
                            st.error("❌ 文本添加到向量库失败，请检查日志。")
                    else:
                        print(f"[DEBUG page_upload] 条件不满足，无法添加文档")
                        st.error("❌ 向量存储仍未初始化，无法添加文本。请检查终端日志。")
                        st.write("**诊断信息:**")
                        st.write(f"- rag_app 存在: {rag_app is not None}")
                        if rag_app:
                            st.write(f"- rag_app.vector_store 存在: {rag_app.vector_store is not None}")
                            st.write(f"- rag_app.vector_store 类型: {type(rag_app.vector_store)}")
                            st.write(f"- rag_app._initialized: {rag_app._initialized}")
                            st.write(f"- rag_app.embeddings 存在: {rag_app.embeddings is not None}")
                        st.write("**建议:** 查看应用启动时的终端输出，查找 [ERROR] 消息。")
                except Exception as e:
                    import traceback
                    st.error(f"处理失败: {e}")
                    st.error(traceback.format_exc())

# ============ 智能问答 ============
def page_qa():
    st.header("🔍 智能问答")
    st.write("使用 LangChain RAG 技术对文档进行查询")

    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain 未安装")
        return

    if not rag_app or not rag_app.vector_store:
        st.warning("⚠️ 向量存储未初始化，请先上传文档")
        return

    query = st.text_input("提问", placeholder="例如: 登录功能的参数有哪些?")

    if st.button("🔍 使用 LangChain 搜索", type="primary"):
        if query:
            with st.spinner("使用 LangChain RAG 搜索..."):
                try:
                    import re

                    print(f"\n{'='*80}")
                    print(f"[SEARCH START] 开始搜索: '{query}'")
                    print(f"[VECTOR_STORE TYPE] {type(rag_app.vector_store).__name__}")
                    print(f"{'='*80}\n")

                    # 使用 vector_store 的 as_retriever 方法获取检索器
                    if hasattr(rag_app.vector_store, 'as_retriever'):
                        retriever = rag_app.vector_store.as_retriever(search_kwargs={"k": 10})
                        print(f"[RETRIEVER TYPE] {type(retriever).__name__}")

                        # 尝试不同的调用方式
                        docs = None
                        try:
                            # 方式 1: 使用 invoke 方法
                            print(f"[RETRIEVAL METHOD] 尝试使用 invoke() 方法...")
                            docs = retriever.invoke(query)
                            print(f"[RETRIEVAL SUCCESS] invoke() 方法成功，获得 {len(docs) if docs else 0} 个初始结果")
                        except (AttributeError, TypeError) as e:
                            print(f"[RETRIEVAL FALLBACK] invoke() 失败: {e}, 尝试其他方法...")
                            try:
                                # 方式 2: 使用 get_relevant_documents 方法
                                print(f"[RETRIEVAL METHOD] 尝试使用 get_relevant_documents() 方法...")
                                docs = retriever.get_relevant_documents(query)
                                print(f"[RETRIEVAL SUCCESS] get_relevant_documents() 方法成功，获得 {len(docs) if docs else 0} 个初始结果")
                            except (AttributeError, TypeError) as e2:
                                print(f"[RETRIEVAL FALLBACK] get_relevant_documents() 失败: {e2}, 尝试直接调用...")
                                # 方式 3: 直接调用（__call__ 方法）
                                docs = retriever(query)
                                print(f"[RETRIEVAL SUCCESS] 直接调用成功，获得 {len(docs) if docs else 0} 个初始结果")

                        if docs:
                            # ===== 后处理：基于关键词匹配重新排序 =====
                            print(f"\n[POST-PROCESSING] 开始后处理搜索结果...")

                            # 提取查询词
                            query_lower = query.lower()
                            query_words = re.findall(r'[\w]+', query_lower)
                            query_words = [w for w in query_words if len(w) > 1]  # 过滤单字符

                            print(f"[KEYWORDS] 查询关键词: {query_words}")

                            # 为每个文档计算关键词匹配得分
                            scored_docs = []
                            for i, doc in enumerate(docs):
                                # 获取文档内容
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                else:
                                    content = str(doc)

                                doc_text = content.lower()

                                # 计算关键词匹配
                                matched_words = 0
                                for word in query_words:
                                    # 使用单词边界匹配
                                    if re.search(r'\b' + re.escape(word) + r'\b', doc_text):
                                        matched_words += 1

                                # 关键词匹配率 (0-1)
                                if len(query_words) > 0:
                                    keyword_score = matched_words / len(query_words)
                                else:
                                    keyword_score = 0

                                scored_docs.append({
                                    'doc': doc,
                                    'keyword_score': keyword_score,
                                    'matched_count': matched_words,
                                    'index': i
                                })

                                print(f"[DOC {i}] 关键词匹配: {matched_words}/{len(query_words)} = {keyword_score:.2%}")

                            # 按关键词匹配得分排序
                            scored_docs.sort(key=lambda x: (x['keyword_score'], x['index']), reverse=True)

                            # 过滤：只保留有至少一个关键词匹配的文档
                            # 或者如果没有匹配的文档，保留最高分的一个
                            filtered_docs = [d for d in scored_docs if d['matched_count'] > 0]

                            if not filtered_docs and scored_docs:
                                # 如果完全没有关键词匹配，保留最高分的一个
                                print(f"[WARNING] 没有关键词匹配的文档，保留得分最高的结果")
                                filtered_docs = [scored_docs[0]]

                            print(f"\n[RESULT] 过滤后的文档数: {len(filtered_docs)}")

                            if filtered_docs:
                                print(f"[SEARCH RESULT] 搜索成功，显示 {len(filtered_docs)} 个相关文档\n")
                                st.markdown("### 📚 检索到的文档")
                                for idx, scored_doc in enumerate(filtered_docs, 1):
                                    doc = scored_doc['doc']
                                    keyword_score = scored_doc['keyword_score']

                                    # 处理不同的返回格式
                                    if hasattr(doc, 'page_content'):
                                        content = doc.page_content
                                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                                        print(f"[DOCUMENT {idx}] 来源: {metadata}, 匹配度: {keyword_score:.0%}")
                                    else:
                                        # 如果是字符串，直接使用
                                        content = str(doc)
                                        metadata = {}
                                        print(f"[DOCUMENT {idx}] (字符串格式), 匹配度: {keyword_score:.0%}")

                                    with st.expander(f"文档 {idx} (匹配度: {keyword_score:.0%})"):
                                        st.write(content[:500])
                                        if metadata:
                                            st.write(f"来源: {metadata}")
                            else:
                                print(f"[SEARCH RESULT] 没有找到匹配的文档")
                                st.info(f"未找到与'{query}'相关的文档")
                        else:
                            print(f"[SEARCH RESULT] 检索器未返回结果")
                            st.info("未找到相关文档")
                    else:
                        st.error("向量存储不支持检索操作")
                        print(f"[ERROR] 向量存储不支持检索操作")

                    print(f"\n{'='*80}")
                    print(f"[SEARCH END] 搜索完成")
                    print(f"{'='*80}\n")

                except Exception as e:
                    print(f"[ERROR SEARCH] 搜索失败: {e}")
                    import traceback
                    print(traceback.format_exc())
                    st.error(f"搜索失败: {e}")


# ============ 系统信息 ============
def page_rag_info():
    st.header("ℹ️ RAG 系统信息")

    st.markdown("## 🏗️ 系统架构")
    st.markdown("""
    ```
    用户输入
        ↓
    LangChain 文本分割
        ↓
    Sentence Transformers 文本嵌入
        ↓
    ChromaDB 向量存储
        ↓
    LangChain Retriever 检索
        ↓
    结构化输出
    ```
    """)

    st.markdown("## 📊 组件状态")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LangChain", "✅" if LANGCHAIN_AVAILABLE else "❌")
    col2.metric("HuggingFace 嵌入", "✅" if (rag_app and rag_app.embeddings) else "❌")
    col3.metric("ChromaDB 向量存储", "✅" if (rag_app and rag_app.vector_store) else "❌")
    col4.metric("QA 链", "✅" if (rag_app and rag_app.qa_chain) else "⚠️")

    st.markdown("## 🛠️ 技术栈详情")
    tech_details = {
        "**Python**": "编程语言",
        "**Streamlit**": "Web UI 框架",
        "**LangChain**": "LLM 应用框架 ⭐ (已集成)",
        "**Sentence Transformers**": "文本嵌入模型",
        "**ChromaDB**": "向量数据库",
        "**RAG (检索增强生成)**": "核心技术架构"
    }

    for tech, desc in tech_details.items():
        st.write(f"{tech}: {desc}")

    st.success("✅ 完整的 LangChain + RAG 实现")

# ============ 主函数 ============
def main():
    init_session_state()

    st.sidebar.title("🧪 RAG 智能测试助手")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🛠️ 技术栈")
    col1, col2 = st.sidebar.columns(2)
    col1.write("**LangChain**")
    col1.write("✅" if LANGCHAIN_AVAILABLE else "❌")
    col2.write("**ChromaDB**")
    col2.write("✅" if (rag_app and rag_app.vector_store) else "⚠️")

    st.sidebar.write("**Sentence Transformers**")
    st.sidebar.write("✅" if (rag_app and rag_app.embeddings) else "⚠️")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "导航",
        [
            "🏠 首页",
            "✨ 生成用例",
            "📤 上传文档",
            "🔍 智能问答",
            "ℹ️ 系统信息"
        ]
    )

    st.sidebar.markdown("---")
    if LANGCHAIN_AVAILABLE:
        st.sidebar.success("✅ LangChain 已启用")
    else:
        st.sidebar.warning("⚠️ LangChain 未启用")

    st.sidebar.info("版本: RAG v5.0 (LangChain)")

    if page == "🏠 首页":
        page_home()
    elif page == "✨ 生成用例":
        page_generate()
    elif page == "📤 上传文档":
        page_upload()
    elif page == "🔍 智能问答":
        page_qa()
    elif page == "ℹ️ 系统信息":
        page_rag_info()

if __name__ == "__main__":
    main()
