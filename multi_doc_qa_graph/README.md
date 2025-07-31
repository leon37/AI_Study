# 📚 Multi-Format Document QA with LangGraph

基于 LangChain + LangGraph 构建的多格式文档问答系统，支持 TXT、PDF、DOCX、Markdown 等格式上传，自动切分并存入向量数据库 FAISS，实现用户自然语言提问与文档内容匹配问答。

---

## 🚀 项目亮点

- ✅ 支持多种文档格式（.txt、.pdf、.docx、.md）
- ✅ 使用 `RecursiveCharacterTextSplitter` 自动分块
- ✅ 使用 `OpenAI Embeddings` 进行向量化
- ✅ 基于 `FAISS` 构建本地向量检索引擎
- ✅ 使用 `LangGraph` 管理 LLM 推理流程
- ✅ 具备简单 RAG（Retrieval-Augmented Generation）能力

---

## 📁 项目结构

multi_doc_qa_graph/
├── main.py # 项目主入口，运行聊天循环
├── loader.py # 文档加载器，自动识别文件格式
├── splitter.py # 文档切分模块
├── retriever.py # FAISS 检索器封装
├── prompt.py # Prompt 构建模块
├── tools.py # 自定义 LangChain 工具（如计算器、外号查询等）
├── graph_builder.py # LangGraph 构建逻辑
├── test_docs/ # 存放各类测试文档
│ ├── sample.txt
│ ├── example.docx
│ └── guide.md
├── requirements.txt # 所有依赖包
└── README.md # 项目说明文档

## 🧪 运行示例

```bash
$ python main.py

Hello
user: 第二阶段应该做什么？
assistant: 第二阶段是对多文档问答系统进行整合，建议构建一个完整的 LangGraph DAG，实现输入问题、检索上下文、生成回答等阶段串联。
```

## 🧠 项目原理简述
文档加载：使用 Unstructured 系列 Loader 解析各种格式；

文档切分：用 RecursiveCharacterTextSplitter 分成更适合嵌入的小块；

向量化：使用 text-embedding-3-small 生成文本向量；

存储：使用 FAISS 存储向量并支持快速近邻检索；

Prompt 构建：注入上下文构造系统提示词；

LangGraph：使用图结构管理 RAG 执行流程，支持中断、回溯与多节点逻辑；

## 👨‍💻 作者
该项目为 LangChain & LangGraph 学习计划 Day1–Day14 实践项目，由 @leon37 基于真实需求自主设计实现。