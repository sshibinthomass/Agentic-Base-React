import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { MODEL_OPTIONS, USE_CASES } from "./constants";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { escapeHtml, renderMarkdown } from "./utils/markdown";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function App() {
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resetting, setResetting] = useState(false);
  const [sessionId, setSessionId] = useState("default");
  const [provider, setProvider] = useState("groq");
  const [model, setModel] = useState(() => MODEL_OPTIONS.groq[0].value);
  const [useCase, setUseCase] = useState(() => USE_CASES[0]?.value ?? "basic_chatbot");
  const [backendStatus, setBackendStatus] = useState("checking");
  const [backendStatusMessage, setBackendStatusMessage] = useState("Checking backend...");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const toolStatusEventSourceRef = useRef(null);
  const [toolStatusStreamKey, setToolStatusStreamKey] = useState(0);
  const [latestToolCall, setLatestToolCall] = useState(null);
  const [toolCallHistory, setToolCallHistory] = useState([]);
  const [toolStatusComplete, setToolStatusComplete] = useState(true);

  const normalizeToolEntry = (entry) => {
    if (!entry) return null;
    if (typeof entry === "string") {
      return {
        id: null,
        name: entry,
        timestamp: "",
        args: {},
        response: "",
        duration_ms: null,
      };
    }
    if (typeof entry !== "object") return null;
    const name =
      typeof entry.name === "string"
        ? entry.name
        : entry.name !== undefined
        ? String(entry.name)
        : "Unknown tool";
    const timestamp = typeof entry.timestamp === "string" ? entry.timestamp : "";
    const args =
      entry.args && typeof entry.args === "object"
        ? entry.args
        : {};
    const response =
      typeof entry.response === "string"
        ? entry.response
        : entry.response !== undefined
        ? JSON.stringify(entry.response)
        : "";
    const duration =
      typeof entry.duration_ms === "number" ? entry.duration_ms : null;
    const id =
      typeof entry.id === "string"
        ? entry.id
        : entry.id !== undefined
        ? String(entry.id)
        : null;
    return {
      id,
      name,
      timestamp,
      args,
      response,
      duration_ms: duration,
    };
  };

  const normalizeToolEntries = (value) => {
    if (!Array.isArray(value)) return [];
    return value
      .map((entry) => normalizeToolEntry(entry))
      .filter((entry) => entry && entry.name);
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const existing = localStorage.getItem("chat_session_id");
    if (existing) {
      setSessionId(existing);
    } else {
      const id =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `session-${Date.now()}`;
      localStorage.setItem("chat_session_id", id);
      setSessionId(id);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`, {
          method: "GET",
        });

        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }

        const data = await response.json().catch(() => ({}));

        if (!isMounted) return;
        setBackendStatus("online");
        setBackendStatusMessage(
          typeof data === "object" && data !== null
            ? data.status ?? "Online"
            : "Online"
        );
      } catch (err) {
        if (!isMounted) return;
        setBackendStatus("offline");
        setBackendStatusMessage(
          err instanceof Error ? err.message : "Unable to reach backend"
        );
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 15000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleProviderChange = (nextProvider) => {
    setProvider(nextProvider);
    const nextModel = MODEL_OPTIONS[nextProvider]?.[0]?.value ?? "";
    setModel(nextModel);
  };

  const handleModelChange = (nextModel) => {
    setModel(nextModel);
  };

  useEffect(() => {
    if (!sessionId) return;
    const params = new URLSearchParams({
      session_id: sessionId || "default",
      use_case: useCase,
    });
    const eventSource = new EventSource(`${BACKEND_URL}/tool-status/stream?${params.toString()}`);
    toolStatusEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLatestToolCall(normalizeToolEntry(data?.latest_tool_call));
        setToolCallHistory(normalizeToolEntries(data?.tool_calls));
        setToolStatusComplete(Boolean(data?.completed));
      } catch (err) {
        // Ignore malformed events
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      if (toolStatusEventSourceRef.current === eventSource) {
        toolStatusEventSourceRef.current = null;
      }
      setTimeout(() => setToolStatusStreamKey((prev) => prev + 1), 1500);
    };

    return () => {
      eventSource.close();
      if (toolStatusEventSourceRef.current === eventSource) {
        toolStatusEventSourceRef.current = null;
      }
    };
  }, [sessionId, useCase, BACKEND_URL, toolStatusStreamKey]);

  const resetConversation = async () => {
    setError("");
    setResetting(true);
    setLatestToolCall(null);
    setToolCallHistory([]);
    setToolStatusComplete(true);
    try {
      const res = await fetch(`${BACKEND_URL}/chat/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId || "default",
          use_case: useCase,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Unable to clear conversation.");
      }
      setConversation([]);
    } catch (err) {
      setError(err.message || "Failed to clear conversation.");
    } finally {
      setResetting(false);
    }
  };

  const handleUseCaseChange = async (nextUseCase) => {
    if (nextUseCase === useCase) return;
    setUseCase(nextUseCase);
    await resetConversation();
  };

  async function handleSubmitMessage(content) {
    setError("");

    const trimmed = content.trim();
    if (!trimmed) return;

    const userMessage = {
      text: trimmed,
      rendered: escapeHtml(trimmed),
      isUser: true,
    };
    setConversation((prev) => [...prev, userMessage]);
    setLoading(true);
    setLatestToolCall(null);
    setToolCallHistory([]);
    setToolStatusComplete(false);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          provider,
          selected_llm: model,
          use_case: useCase,
          session_id: sessionId || "default",
        }),
      });

      const payload = await response
        .json()
        .catch(async () => ({ detail: await response.text() }));

      if (!response.ok) {
        const detail =
          typeof payload?.detail === "string"
            ? payload.detail
            : "Chatbot backend returned an error.";
        throw new Error(detail);
      }

      const botText = payload?.response ?? "No response";
      const rendered = renderMarkdown(botText);
      setConversation((prev) => [
        ...prev,
        { text: botText, rendered, isUser: false },
      ]);
      const normalizedCalls = normalizeToolEntries(payload?.tool_calls);
      setToolCallHistory(normalizedCalls);
      const latest =
        normalizeToolEntry(payload?.latest_tool_call) ??
        (normalizedCalls.length ? normalizedCalls[normalizedCalls.length - 1] : null);
      setLatestToolCall(latest);
      setToolStatusComplete(true);
    } catch (err) {
      let message = err.message || "Something went wrong";
      if (message.includes("Chatbot not initialized")) {
        message =
          "Chatbot service is not ready yet. Please ensure the backend is running with a valid GROQ_API_KEY.";
      }
      setError(message);
      setToolStatusComplete(true);
      setLatestToolCall(null);
      setToolCallHistory([]);
    } finally {
      setLoading(false);
    }
  }

  const availableModels = MODEL_OPTIONS[provider] ?? [];
  const activeUseCaseLabel =
    USE_CASES.find((option) => option.value === useCase)?.label || "Chat";

  return (
    <div className={`app ${sidebarOpen ? "app--sidebar-open" : "app--sidebar-closed"}`}>
      {sidebarOpen && (
        <Sidebar
          provider={provider}
          model={model}
          models={availableModels}
          useCase={useCase}
          useCases={USE_CASES}
          onProviderChange={handleProviderChange}
          onModelChange={handleModelChange}
          onUseCaseChange={handleUseCaseChange}
          backendUrl={BACKEND_URL}
          backendStatus={backendStatus}
          backendStatusMessage={backendStatusMessage}
        />
      )}

      <ChatWindow
        conversation={conversation}
        onSubmit={handleSubmitMessage}
        onClear={resetConversation}
        loading={loading}
        resetting={resetting}
        error={error}
        useCaseLabel={activeUseCaseLabel}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        latestToolCall={latestToolCall}
        toolCallHistory={toolCallHistory}
        toolStatusComplete={toolStatusComplete}
      />
    </div>
  );
}


