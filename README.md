# rag_fastapi_simplified

一个基于原 Java RAG 核心逻辑简化迁移的 Python 版本，包含：

- FastAPI 后端
- LangChain Agent 架构
- Milvus 向量库
- Ollama Embedding 与 Ollama Rerank
- 多路检索（意图定向 + 全局向量补充）
- 意图识别与引导式歧义澄清
- 问题改写与拆分
- 简易前端页面

## 1. 核心对齐点（对应 Java 项目策略）

- 主流程：改写拆分 -> 意图识别 -> 歧义引导 -> 多通道检索 -> 去重 -> rerank -> 生成回答
- 检索通道策略：
  - IntentDirectedSearchChannel：按意图 collection 定向检索，topK 倍率 2
  - VectorGlobalSearchChannel：低置信度或无意图时启用全局补充，topK 倍率 3
- 后处理策略：
  - Deduplication：跨通道去重，保留更高分
- Rerank：通过 Ollama 服务重排，返回最终 TopK，并把 rerank score 透传到前端
- 分块策略：
  - fixed_size：带 overlap 与边界对齐
  - structure_aware：Markdown/段落/代码块友好切分
- 引导策略：
  - 单子问题且同名主题跨系统高分接近时，触发澄清选项

## 2. 目录结构

- app/main.py: FastAPI 入口
- app/api/routes_chat.py: 问答接口
- app/api/routes_ingest.py: 文档入库接口
- app/services/: RAG 各服务实现
- app/data/intent_tree.sample.json: 示例意图树
- frontend/: 简易网页

## 3. 安装与运行

```bash
cd /userhome/cs/u3651279/workspace/ragent/rag_fastapi_simplified
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

如需自定义 collection schema，请修改 `.env` 中的 `COLLECTION_CONFIG_PATH` 指向你的 JSON 配置文件。

后端健康检查：

- http://127.0.0.1:8080/health

## 3.1 Reflex 前端

使用 Python Reflex 取代静态页面：

```bash
cd /userhome/cs/u3651279/workspace/ragent/rag_fastapi_simplified/frontend_reflex
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 如果后端不是 8188 端口，请调整 API_BASE_URL
API_BASE_URL=http://127.0.0.1:8080 reflex run
```

默认访问：

- http://127.0.0.1:3000/

## 3.2 Streamlit 前端

使用 Streamlit 搭建交互式控制台（Chat / Upload / Milvus）：

```bash
cd /userhome/cs/u3651279/workspace/ragent/rag_fastapi_simplified/frontend_streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
API_BASE_URL=http://127.0.0.1:8080 streamlit run app.py
```

默认访问：

- http://127.0.0.1:8501/

## 4. 接口

### 4.1 文档入库

- `POST /api/v1/ingest/file`
- form-data:
  - `file`: txt/md/pdf
  - `collection_name`: 目标 Milvus collection
  - `chunk_mode`: `structure_aware` 或 `fixed_size`

### 4.2 问答

- `POST /api/v1/chat`
- body:

```json
{
  "question": "OA系统的数据安全策略是什么？",
  "top_k": 8,
  "history": []
}
```

### 4.3 Milvus 集合查看

- `GET /api/v1/milvus/collections`

## 5. 模型建议

- Chat: Qwen3 Instruct（OpenAI 兼容接口）
- Embedding: Qwen3 Embedding
- Rerank: Qwen3 Reranker（通过 Ollama 服务提供）

如果你使用 Ollama/vLLM，请把 `.env` 中的 `CHAT_BASE_URL`、`CHAT_MODEL`，以及 rerank 相关配置改成你的服务配置。

## 6. 说明

- 这是简化工程，重点复刻 Java 核心策略，不包含原项目中数据库管理后台、MCP 执行链、SSE 流式控制等完整能力。
- 若需要，我可以在这套基础上继续补：
  - SSE 流式输出
  - 多租户会话记忆
  - 意图树后台管理
  - Redis 缓存与并发限流
