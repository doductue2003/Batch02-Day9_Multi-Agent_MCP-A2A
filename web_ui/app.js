const services = [
  { name: "Registry", port: 10000, url: "http://localhost:10000/health", type: "json" },
  { name: "Customer", port: 10100, url: "http://localhost:10100/.well-known/agent.json", type: "card" },
  { name: "Law", port: 10101, url: "http://localhost:10101/.well-known/agent.json", type: "card" },
  { name: "Tax", port: 10102, url: "http://localhost:10102/.well-known/agent.json", type: "card" },
  { name: "Compliance", port: 10103, url: "http://localhost:10103/.well-known/agent.json", type: "card" },
];

const demoSteps = [
  ["Customer Agent", "Receives the user question and delegates legal analysis."],
  ["Law Agent", "Analyzes contract liability and decides which specialists are needed."],
  ["Tax Agent", "Reviews evasion, IRS enforcement, civil penalties, and criminal exposure."],
  ["Compliance Agent", "Reviews regulatory risks, SEC/FTC concerns, and governance obligations."],
  ["Privacy Agent", "Reviews consent, data protection, GDPR/CCPA, and breach exposure."],
  ["Aggregate", "Combines specialist analysis into a final answer."],
];

const $ = (selector) => document.querySelector(selector);

const state = {
  onlineCount: 0,
  lastRun: null,
};

function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const rand = (Math.random() * 16) | 0;
    const value = char === "x" ? rand : (rand & 0x3) | 0x8;
    return value.toString(16);
  });
}

function setRunState(label, kind = "idle") {
  const pill = $("#runState");
  pill.textContent = label;
  pill.className = `status-pill ${kind}`;
}

function setOutput(value) {
  $("#responseOutput").textContent = value || "";
}

function addTimelineStep(title, detail) {
  const timeline = $("#timeline");
  const index = timeline.children.length + 1;
  const item = document.createElement("div");
  item.className = "timeline-step";
  item.innerHTML = `
    <div class="step-index">${index}</div>
    <div>
      <p class="step-title"></p>
      <p class="step-detail"></p>
    </div>
  `;
  item.querySelector(".step-title").textContent = title;
  item.querySelector(".step-detail").textContent = detail;
  timeline.appendChild(item);
}

function resetTimeline() {
  $("#timeline").innerHTML = "";
}

function updateMetrics(registryStatus = "Unknown") {
  $("#registryMetric").textContent = registryStatus;
  $("#agentMetric").textContent = String(state.onlineCount);
  $("#lastRunMetric").textContent = state.lastRun || "None";
}

function renderServices(results = []) {
  const list = $("#serviceList");
  list.innerHTML = "";
  services.forEach((service, index) => {
    const result = results[index] || { ok: false };
    const item = document.createElement("div");
    item.className = "service-item";
    item.innerHTML = `
      <span class="service-dot ${result.ok ? "online" : "offline"}"></span>
      <span class="service-name"></span>
      <span class="service-port"></span>
    `;
    item.querySelector(".service-name").textContent = service.name;
    item.querySelector(".service-port").textContent = String(service.port);
    list.appendChild(item);
  });
}

async function checkService(service) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(service.url, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeout);
    if (!response.ok) return { ok: false };
    const data = await response.json();
    return { ok: true, data };
  } catch (error) {
    clearTimeout(timeout);
    return { ok: false, error };
  }
}

async function refreshStatus() {
  const results = await Promise.all(services.map(checkService));
  state.onlineCount = results.filter((result) => result.ok).length;
  renderServices(results);

  const registry = results[0];
  const registryStatus = registry.ok
    ? `OK (${registry.data.agent_count ?? "?"})`
    : "Offline";
  updateMetrics(registryStatus);
}

function buildA2ARequest(question) {
  return {
    id: uuid(),
    jsonrpc: "2.0",
    method: "message/send",
    params: {
      configuration: null,
      message: {
        contextId: null,
        extensions: null,
        kind: "message",
        messageId: uuid(),
        metadata: null,
        parts: [{ kind: "text", metadata: null, text: question }],
        referenceTaskIds: null,
        role: "user",
        taskId: null,
      },
      metadata: null,
    },
  };
}

function collectTextParts(value, output = []) {
  if (!value || typeof value !== "object") return output;
  if (typeof value.text === "string") output.push(value.text);
  if (Array.isArray(value)) {
    value.forEach((item) => collectTextParts(item, output));
    return output;
  }
  Object.values(value).forEach((item) => collectTextParts(item, output));
  return output;
}

function parseA2AResponse(payload) {
  const texts = collectTextParts(payload);
  const unique = [...new Set(texts.map((text) => text.trim()).filter(Boolean))];
  return unique.length ? unique.join("\n\n") : JSON.stringify(payload, null, 2);
}

async function sendQuestion() {
  const endpoint = $("#customerEndpoint").value.trim().replace(/\/$/, "");
  const question = $("#questionInput").value.trim();
  if (!endpoint || !question) {
    setRunState("Missing input", "error");
    setOutput("Customer Agent endpoint and question are required.");
    return;
  }

  resetTimeline();
  setOutput("Sending request...");
  setRunState("Running", "running");
  addTimelineStep("Connect", `Resolving Customer Agent at ${endpoint}`);

  try {
    const cardResponse = await fetch(`${endpoint}/.well-known/agent.json`, {
      cache: "no-store",
    });
    if (!cardResponse.ok) {
      throw new Error(`Agent card returned HTTP ${cardResponse.status}`);
    }
    const card = await cardResponse.json();
    addTimelineStep("Agent Card", `${card.name || "Customer Agent"} ${card.version || ""}`.trim());

    const requestBody = buildA2ARequest(question);
    addTimelineStep("JSON-RPC", "Posting message/send request to the Customer Agent.");
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error?.message || `Request returned HTTP ${response.status}`);
    }

    addTimelineStep("Response", "Received A2A task result.");
    const text = parseA2AResponse(payload);
    setOutput(text);
    state.lastRun = new Date().toLocaleTimeString();
    setRunState("Done", "done");
    updateMetrics($("#registryMetric").textContent);
  } catch (error) {
    addTimelineStep("Stopped", error.message);
    setOutput(
      [
        "Request failed.",
        "",
        error.message,
        "",
        "If the services are running but the browser blocks the request, use test_client.py or add CORS/proxy support.",
        "If the message mentions 429, the OpenRouter free-model quota is exhausted.",
      ].join("\n"),
    );
    setRunState("Error", "error");
  }
}

function runDemo() {
  resetTimeline();
  setRunState("Demo", "running");
  setOutput("Running local demo flow...");

  demoSteps.forEach(([title, detail], index) => {
    window.setTimeout(() => {
      addTimelineStep(title, detail);
      if (index === demoSteps.length - 1) {
        const answer = [
          "Demo response",
          "",
          "Contract: breach may lead to expectation damages, consequential damages, injunctions, and attorney fees depending on the agreement.",
          "",
          "Tax: evasion can trigger back taxes, interest, civil fraud penalties, audits, and potential criminal exposure.",
          "",
          "Compliance and privacy: sharing user data without consent can create FTC, GDPR, CCPA, governance, and class-action risk.",
        ].join("\n");
        setOutput(answer);
        state.lastRun = new Date().toLocaleTimeString();
        setRunState("Done", "done");
        updateMetrics($("#registryMetric").textContent);
      }
    }, index * 220);
  });
}

function bindEvents() {
  $("#refreshStatus").addEventListener("click", refreshStatus);
  $("#sendQuestion").addEventListener("click", sendQuestion);
  $("#runDemo").addEventListener("click", runDemo);
  $("#clearOutput").addEventListener("click", () => {
    resetTimeline();
    setOutput("Ready.");
    setRunState("Idle", "idle");
  });
  $("#copyResponse").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("#responseOutput").textContent);
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
    });
  });
}

bindEvents();
renderServices();
refreshStatus();
