const state = {
  mode: "answer",
  topK: 5,
  messages: [],
  lastSources: [],
  loading: false,
};

const els = {
  chat: document.querySelector("#chat"),
  sources: document.querySelector("#sources"),
  sourceCount: document.querySelector("#sourceCount"),
  form: document.querySelector("#composer"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendBtn"),
  topK: document.querySelector("#topK"),
  topKValue: document.querySelector("#topKValue"),
  lastLatency: document.querySelector("#lastLatency"),
  lastSource: document.querySelector("#lastSource"),
  guardrailState: document.querySelector("#guardrailState"),
  clear: document.querySelector("#clearBtn"),
  segments: Array.from(document.querySelectorAll(".segment")),
  docCount: document.querySelector("#docCount"),
  chunkCount: document.querySelector("#chunkCount"),
  embeddingModel: document.querySelector("#embeddingModel"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswer(value) {
  const lines = String(value)
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return "";

  const listItems = [];
  const paragraphs = [];
  lines.forEach((line) => {
    if (/^[-*]\s+/.test(line)) {
      listItems.push(`<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`);
    } else {
      paragraphs.push(`<p>${escapeHtml(line)}</p>`);
    }
  });

  return `${paragraphs.join("")}${listItems.length ? `<ul>${listItems.join("")}</ul>` : ""}`;
}

function setMode(mode) {
  state.mode = mode;
  els.segments.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
}

function addMessage(role, content, meta = "") {
  state.messages.push({ role, content, meta });
  renderMessages();
}

function renderMessages() {
  if (!state.messages.length) {
    els.chat.innerHTML = `
      <div class="empty-state">
        <div class="scanner" aria-hidden="true"></div>
        <strong>Ask a legal question or inspect retrieved evidence.</strong>
      </div>
    `;
    return;
  }

  els.chat.innerHTML = state.messages
    .map((message) => {
      const avatar = message.role === "user" ? "YOU" : "AI";
      const meta = message.meta ? `<div class="bubble-meta">${escapeHtml(message.meta)}</div>` : "";
      const body = message.role === "assistant" ? formatAnswer(message.content) : escapeHtml(message.content);
      return `
        <article class="message ${message.role}">
          <div class="avatar" aria-hidden="true">${avatar}</div>
          <div class="bubble">${body}${meta}</div>
        </article>
      `;
    })
    .join("");
  els.chat.scrollTop = els.chat.scrollHeight;
}

function sourceLabel(source, index) {
  const metadata = source.metadata || {};
  return metadata.source || `Source ${index + 1}`;
}

function renderSources(sources) {
  state.lastSources = sources || [];
  els.sourceCount.textContent = String(state.lastSources.length);
  if (!state.lastSources.length) {
    els.sources.innerHTML = '<p class="muted">No evidence loaded.</p>';
    return;
  }

  els.sources.innerHTML = state.lastSources
    .map((source, index) => {
      const metadata = source.metadata || {};
      const label = sourceLabel(source, index);
      const meta = `${metadata.type || "unknown"} | chunk ${metadata.chunk_index ?? "n/a"} | ${
        source.source || "unknown"
      } | score ${Number(source.score || 0).toFixed(3)}`;
      return `
        <article class="source-item">
          <button class="source-toggle" type="button" data-index="${index}">
            <span>${index + 1}. ${escapeHtml(label)}</span>
            <span>${index === 0 ? "collapse" : "expand"}</span>
          </button>
          <div class="source-body ${index === 0 ? "" : "hidden"}">
            <div class="source-meta">${escapeHtml(meta)}</div>
            <p>${escapeHtml(source.content || "")}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function setLoading(loading) {
  state.loading = loading;
  els.send.disabled = loading;
  els.send.textContent = loading ? "Wait" : "Send";
  els.guardrailState.textContent = loading ? "Scanning input" : "Ready";
}

async function loadManifest() {
  const response = await fetch("/api/manifest");
  const manifest = await response.json();
  els.docCount.textContent = manifest.document_count ?? "--";
  els.chunkCount.textContent = manifest.chunk_count ?? "--";
  els.embeddingModel.textContent = manifest.embedding_model || "not indexed";
}

async function submitMessage(message) {
  setLoading(true);
  addMessage("user", message);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, mode: state.mode, top_k: state.topK }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }

    const latency = Number(payload.latency || 0).toFixed(2);
    const retrievalSource = payload.retrieval_source || "unknown";
    els.lastLatency.textContent = `${latency}s`;
    els.lastSource.textContent = `source: ${retrievalSource}`;
    els.guardrailState.textContent = retrievalSource === "blocked" ? "Request blocked" : "Ready";

    addMessage("assistant", payload.answer || "No answer returned.", `${retrievalSource} | ${latency}s`);
    renderSources(payload.sources || []);
  } catch (error) {
    addMessage("assistant", error.message || "Request failed.", "error");
    els.guardrailState.textContent = "Error";
  } finally {
    setLoading(false);
  }
}

els.segments.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.topK.addEventListener("input", (event) => {
  state.topK = Number(event.target.value);
  els.topKValue.textContent = String(state.topK);
});

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || state.loading) return;
  els.input.value = "";
  submitMessage(message);
});

els.clear.addEventListener("click", () => {
  state.messages = [];
  state.lastSources = [];
  els.lastLatency.textContent = "idle";
  els.lastSource.textContent = "source: none";
  els.guardrailState.textContent = "Awaiting query";
  renderMessages();
  renderSources([]);
});

els.sources.addEventListener("click", (event) => {
  const button = event.target.closest(".source-toggle");
  if (!button) return;
  const body = button.parentElement.querySelector(".source-body");
  const marker = button.querySelector("span:last-child");
  body.classList.toggle("hidden");
  marker.textContent = body.classList.contains("hidden") ? "expand" : "collapse";
});

loadManifest().catch(() => {
  els.embeddingModel.textContent = "manifest unavailable";
});
renderMessages();
renderSources([]);
