from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


API_BASE_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
API_PREFIX_DEFAULT = os.getenv("API_PREFIX", "/api/v1")


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1c1f2b;
            --accent: #0f6b6d;
            --accent-2: #c04b19;
            --paper: #fffdf7;
            --panel: rgba(255, 255, 255, 0.75);
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 8% 10%, #f5efe6, transparent 45%),
                        radial-gradient(circle at 90% 20%, #e2f2ec, transparent 45%),
                        linear-gradient(120deg, #f7f1e9 0%, #f0f4ef 100%);
            color: var(--ink);
        }

        .hero {
            padding: 18px 18px 6px 18px;
            border-radius: 16px;
            background: var(--panel);
            border: 1px solid rgba(30, 41, 59, 0.08);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }

        .hero h1 {
            font-family: "Space Grotesk", "Noto Sans SC", sans-serif;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }

        .tag {
            display: inline-block;
            margin-top: 8px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(15, 107, 109, 0.12);
            color: var(--accent);
            font-size: 12px;
            font-weight: 600;
        }

        .panel-card {
            background: var(--paper);
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
            padding: 16px;
        }

        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(12, 74, 110, 0.15);
            background: rgba(15, 107, 109, 0.08);
            color: var(--accent);
            font-weight: 600;
            font-size: 12px;
        }

        .chunk-card {
            border: 1px dashed rgba(148, 163, 184, 0.7);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            background: #fffaf0;
        }

        .chunk-score {
            color: var(--accent-2);
            font-weight: 700;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _api_url(base: str, prefix: str, path: str) -> str:
    base = base.rstrip("/")
    prefix = prefix.strip("/")
    path = path.lstrip("/")
    return f"{base}/{prefix}/{path}" if prefix else f"{base}/{path}"


def _request_json(method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
    resp = requests.request(method, url, timeout=420, **kwargs)
    if not resp.ok:
        try:
            payload = resp.json()
            detail = payload.get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(detail)
    return resp.json()


def _load_configured_collections(api_base: str, api_prefix: str) -> List[Dict[str, Any]]:
    url = _api_url(api_base, api_prefix, "milvus/config/collections")
    try:
        data = _request_json("GET", url)
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and item.get("name")]
    return []


def _sort_chunks(chunks: List[Dict[str, Any]], metric: str, order: str) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    def _key(item: Dict[str, Any]) -> Any:
        if metric == "score":
            return float(item.get("score", 0))
        if metric == "intent_id":
            return str(item.get("intent_id", ""))
        if metric == "text_length":
            return len(item.get("text", ""))
        return float(item.get("score", 0))

    reverse = order == "desc"
    return sorted(chunks, key=_key, reverse=reverse)


st.set_page_config(page_title="RAG Agent Streamlit", layout="wide")
_inject_theme()

st.markdown(
    """
    <div class="hero">
        <h1>RAG Agent 控制台</h1>
        <div>Streamlit 前端 · FastAPI + LangChain Agent + Milvus</div>
        <span class="tag">Chat / Upload / Milvus</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("连接配置")
    api_base = st.text_input("API Base URL", value=API_BASE_DEFAULT)
    api_prefix = st.text_input("API Prefix", value=API_PREFIX_DEFAULT)
    st.caption("确保后端已启动，例如 http://127.0.0.1:8080")

    st.divider()
    st.subheader("检索控制")
    top_k = st.number_input("Retrieval TopK", min_value=1, max_value=50, value=8)
    sort_metric = st.selectbox("排序指标", ["score", "intent_id", "text_length"], index=0)
    sort_order = st.selectbox("排序顺序", ["desc", "asc"], index=0)


chat_tab, upload_tab, milvus_tab = st.tabs(["Chat", "Upload", "Milvus 数据库"])

with chat_tab:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    question = st.text_area("问题", placeholder="输入问题，例如：总结一下这篇论文的贡献", height=120)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        do_chat = st.button("发送问题")
    with col2:
        st.markdown(f'<span class="metric-chip">TopK: {int(top_k)}</span>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="metric-chip">排序: {sort_metric}/{sort_order}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if do_chat:
        if not question.strip():
            st.warning("请输入问题再提交。")
        else:
            try:
                payload = {
                    "question": question.strip(),
                    "top_k": int(top_k),
                    "history": st.session_state.chat_history,
                }
                url = _api_url(api_base, api_prefix, "chat")
                data = _request_json("POST", url, json=payload)

                intent_status = data.get("intent_status")
                intent_confidence = data.get("intent_confidence")
                need_rag = data.get("need_rag")
                chat_type = data.get("chat_type")

                st.subheader("意图路由")
                st.write(
                    {
                        "intent_status": intent_status,
                        "intent_confidence": intent_confidence,
                        "need_rag": need_rag,
                        "chat_type": chat_type,
                    }
                )

                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                if data.get("answer"):
                    st.session_state.chat_history.append({"role": "assistant", "content": data.get("answer")})

                st.subheader("改写与拆分")
                st.write(data.get("rewritten_question", ""))
                sub_questions = data.get("sub_questions", [])
                if sub_questions:
                    st.caption("子问题")
                    st.write(sub_questions)

                guidance_prompt = data.get("guidance_prompt")
                if guidance_prompt:
                    st.info(guidance_prompt)

                st.subheader("回答")
                st.write(data.get("answer", ""))

                intents = data.get("intents", [])
                if intents:
                    st.subheader("意图识别")
                    st.dataframe(intents, use_container_width=True)

                if intent_status == "rag" and need_rag:
                    chunks = data.get("chunks", [])
                    if chunks:
                        st.subheader("命中片段")
                        chunks = _sort_chunks(chunks, sort_metric, sort_order)
                        for item in chunks:
                            rerank_score = float(item.get("rerank_score") or item.get("score", 0))
                            retrieval_score = float(item.get("retrieval_score") or item.get("score", 0))
                            intent_id = item.get("intent_id", "")
                            text = item.get("text", "")
                            st.markdown(
                                f"""
                                <div class="chunk-card">
                                    <div class="chunk-score">rerank_score={rerank_score:.4f} · retrieval_score={retrieval_score:.4f} · intent={intent_id}</div>
                                    <div>{text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("没有命中片段。")
                elif intent_status == "clarify":
                    st.warning("意图不明确，已停在意图分析阶段。请补充需求。")
            except Exception as exc:
                st.error(f"请求失败: {exc}")

with upload_tab:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("文档入库")
    configured_collections = _load_configured_collections(api_base, api_prefix)
    if configured_collections:
        collection_options = [item["name"] for item in configured_collections]
        collection_name = st.selectbox("Collection", collection_options, index=0)
        selected_item = next(
            (item for item in configured_collections if item["name"] == collection_name),
            None,
        )
        if selected_item and selected_item.get("description"):
            st.caption(selected_item["description"])
        manual_override = st.checkbox("手动输入 Collection", value=False)
        if manual_override:
            manual_name = st.text_input("手动 Collection", value=collection_name)
            if manual_name.strip():
                collection_name = manual_name.strip()
    else:
        collection_name = st.text_input("Collection", value="rag_default_store")
    chunk_mode = st.selectbox(
        "Chunk Mode",
        [
            "recursive",
            "character",
            "token",
            "markdown_header",
            "html_header",
            "python_code",
        ],
        index=0,
    )
    bm25_language = st.selectbox(
        "BM25 Language",
        ["en", "zh", "ja", "ko"],
        index=0,
    )
    upload_types = [
        "txt",
        "md",
        "pdf",
        "csv",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "bmp",
        "gif",
        "webp",
    ]
    uploaded_file = st.file_uploader("上传文件", type=upload_types)
    upload_btn = st.button("上传并切块入库")
    st.markdown("</div>", unsafe_allow_html=True)

    if upload_btn:
        if not uploaded_file:
            st.warning("请先选择文件。")
        else:
            try:
                url = _api_url(api_base, api_prefix, "ingest/file")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = _request_json(
                    "POST",
                    url,
                    files=files,
                    data={
                        "collection_name": collection_name.strip() or "rag_default_store",
                        "chunk_mode": chunk_mode,
                        "bm25_language": bm25_language,
                    },
                )
                st.success(
                    f"入库成功: collection={data.get('collection_name')} chunks={data.get('chunk_count')}"
                )
            except Exception as exc:
                st.error(f"上传失败: {exc}")

with milvus_tab:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Milvus 集合")
    refresh_btn = st.button("刷新集合列表")
    if refresh_btn:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        url = _api_url(api_base, api_prefix, "milvus/collections")
        data = _request_json("GET", url)
        if data:
            st.dataframe(data, use_container_width=True)
        else:
            st.info("暂无集合。")

        if data:
            collections = [item.get("name") for item in data if item.get("name")]
            st.subheader("实体浏览")
            selected = st.selectbox("选择 Collection", collections)
            if selected:
                schema_url = _api_url(api_base, api_prefix, f"milvus/collections/{selected}/schema")
                schema = _request_json("GET", schema_url)
                field_names = [field.get("name") for field in schema if field.get("name")]
                viewable_fields = [
                    field.get("name")
                    for field in schema
                    if field.get("name") and field.get("viewable", True)
                ]
                default_fields = [name for name in viewable_fields if name != "id"] or viewable_fields
                fields = st.multiselect("输出字段", viewable_fields, default=default_fields)
                hidden_fields = [
                    field.get("name")
                    for field in schema
                    if field.get("name") and not field.get("viewable", True)
                ]
                if hidden_fields:
                    st.caption(f"已隐藏向量字段: {', '.join(hidden_fields)}")
                col_left, col_right = st.columns(2)
                with col_left:
                    entity_limit = st.number_input("每页数量", min_value=1, max_value=200, value=50)
                with col_right:
                    entity_offset = st.number_input("Offset", min_value=0, max_value=100000, value=0)

                if st.button("加载实体"):
                    if not fields:
                        st.warning("请至少选择一个字段。")
                    else:
                        entities_url = _api_url(
                            api_base,
                            api_prefix,
                            f"milvus/collections/{selected}/entities",
                        )
                        entities = _request_json(
                            "GET",
                            entities_url,
                            params={
                                "limit": int(entity_limit),
                                "offset": int(entity_offset),
                                "output_fields": ",".join(fields),
                            },
                        )
                        if entities:
                            st.dataframe(entities, use_container_width=True)
                        else:
                            st.info("没有返回实体。")
    except Exception as exc:
        st.error(f"获取集合失败: {exc}")
