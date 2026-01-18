import React, { useState, useEffect } from "react";

export function Sidebar({
  provider,
  model,
  models,
  useCase,
  useCases,
  sessionId,
  sessions,
  onProviderChange,
  onModelChange,
  onUseCaseChange,
  onSessionChange,
  onCreateSession,
  onRefreshSessions,
  backendUrl,
  backendStatus,
  backendStatusMessage,
}) {
  const [showSessionList, setShowSessionList] = useState(false);
  const activeUseCaseLabel =
    useCases.find((option) => option.value === useCase)?.label || "Chat";

  const statusLabel = (() => {
    switch (backendStatus) {
      case "online":
        return "Backend Online";
      case "offline":
        return "Backend Offline";
      case "checking":
      default:
        return "Checking Backend";
    }
  })();

  return (
    <aside className="sidebar">
      <h1 className="sidebar__title">{activeUseCaseLabel}</h1>
      <div className="sidebar__form">
        <label className="sidebar__label">
          Session
          <div style={{ display: "flex", gap: "4px", marginTop: "4px" }}>
            <select
              value={sessionId}
              onChange={(event) => onSessionChange(event.target.value)}
              className="sidebar__select"
              style={{ flex: 1 }}
            >
              {sessions && sessions.length > 0 ? (
                sessions.map((session) => (
                  <option key={session.session_id} value={session.session_id}>
                    {session.session_id.length > 20
                      ? `${session.session_id.substring(0, 20)}...`
                      : session.session_id}
                    {session.message_count > 0 ? ` (${session.message_count} msgs)` : ""}
                  </option>
                ))
              ) : (
                <option value={sessionId || "default"}>
                  {sessionId || "default"}
                </option>
              )}
            </select>
            <button
              onClick={onCreateSession}
              className="sidebar__button"
              style={{
                padding: "4px 8px",
                fontSize: "12px",
                cursor: "pointer",
                backgroundColor: "#4CAF50",
                color: "white",
                border: "none",
                borderRadius: "4px",
              }}
              title="Create New Session"
            >
              +
            </button>
            <button
              onClick={onRefreshSessions}
              className="sidebar__button"
              style={{
                padding: "4px 8px",
                fontSize: "12px",
                cursor: "pointer",
                backgroundColor: "#2196F3",
                color: "white",
                border: "none",
                borderRadius: "4px",
              }}
              title="Refresh Sessions"
            >
              ↻
            </button>
          </div>
        </label>

        <label className="sidebar__label">
          Use Case
          <select
            value={useCase}
            onChange={(event) => onUseCaseChange(event.target.value)}
            className="sidebar__select"
          >
            {useCases.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="sidebar__label">
          Provider
          <select
            value={provider}
            onChange={(event) => onProviderChange(event.target.value)}
            className="sidebar__select"
          >
            <option value="groq">Groq</option>
            <option value="openai">OpenAI</option>
            <option value="gemini">Gemini</option>
            <option value="ollama">Ollama</option>
            <option value="anthropic">Anthropic</option>
          </select>
        </label>

        <label className="sidebar__label">
          Model
          <select
            value={model}
            onChange={(event) => onModelChange(event.target.value)}
            className="sidebar__select"
          >
            {models.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="sidebar__footer">
        <span className={`sidebar__status sidebar__status--${backendStatus}`}>
          <span className="sidebar__status-indicator" />
          {statusLabel}
        </span>
        <div className="sidebar__backend-url">
          <code>{backendUrl}</code>
        </div>
        {backendStatusMessage && (
          <div className="sidebar__footer-message">{backendStatusMessage}</div>
        )}
      </div>
    </aside>
  );
}
