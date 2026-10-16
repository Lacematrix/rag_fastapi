const uploadForm = document.getElementById("upload-form");
const uploadResult = document.getElementById("upload-result");
const askBtn = document.getElementById("ask-btn");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  uploadResult.textContent = "上传中...";

  const file = document.getElementById("file-input").files[0];
  const collection = document.getElementById("collection").value.trim();
  const chunkMode = document.getElementById("chunk-mode").value;

  if (!file) {
    uploadResult.textContent = "请先选择文件";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection_name", collection || "rag_default_store");
  formData.append("chunk_mode", chunkMode);

  try {
    const resp = await fetch("/api/v1/ingest/file", {
      method: "POST",
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "上传失败");
    }
    uploadResult.textContent = `入库成功: collection=${data.collection_name}, chunks=${data.chunk_count}`;
  } catch (err) {
    uploadResult.textContent = `上传失败: ${err.message}`;
  }
});

askBtn.addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const topk = Number(document.getElementById("topk").value || 8);

  if (!question) {
    return;
  }

  document.getElementById("rewrite-box").textContent = "处理中...";
  document.getElementById("answer-box").textContent = "";
  document.getElementById("chunks-box").innerHTML = "";

  try {
    const resp = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topk }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "提问失败");
    }

    document.getElementById("rewrite-box").textContent =
      `rewritten: ${data.rewritten_question}\nsub_questions: ${JSON.stringify(data.sub_questions, null, 2)}`;
    document.getElementById("answer-box").textContent = data.answer;

    const chunksBox = document.getElementById("chunks-box");
    if (!data.chunks || data.chunks.length === 0) {
      chunksBox.textContent = "没有命中片段";
      return;
    }

    chunksBox.innerHTML = data.chunks
      .map(
        (c) =>
          `<div class="chunk-item"><div class="chunk-score">rerank_score=${formatScore(c.rerank_score ?? c.score)} · retrieval_score=${formatScore(c.retrieval_score ?? c.score)} · intent=${escapeHtml(c.intent_id)}</div><div>${escapeHtml(c.text)}</div></div>`
      )
      .join("");
  } catch (err) {
    document.getElementById("answer-box").textContent = `请求失败: ${err.message}`;
  }
});

function formatScore(value) {
  const number = Number(value);
  if (Number.isNaN(number)) {
    return "0.0000";
  }
  return number.toFixed(4);
}

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
