import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

function CloudMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M7 18h10a4 4 0 0 0 .8-7.92 5.5 5.5 0 0 0-10.76 1.1A3.5 3.5 0 0 0 7 18Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 12h13M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TagIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 12.5V5a1 1 0 0 1 1-1h7.5a1 1 0 0 1 .7.3l6.5 6.5a1 1 0 0 1 0 1.4l-7 7a1 1 0 0 1-1.4 0l-6.5-6.5a1 1 0 0 1-.3-.7Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="8.5" cy="8.5" r="1.3" fill="currentColor" />
    </svg>
  );
}

function SyncIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 12a8 8 0 0 1 13.7-5.6L20 8M20 4v4h-4M20 12a8 8 0 0 1-13.7 5.6L4 16M4 20v-4h4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CodeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="m8 8-4 4 4 4M16 8l4 4-4 4M13 5l-2 14"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const EXAMPLES = [
  { label: "How much does the Pro plan cost?", hint: "Plans & pricing", icon: <TagIcon /> },
  { label: "My files aren't syncing on my laptop", hint: "Troubleshooting", icon: <SyncIcon /> },
  { label: "Which plans include API access?", hint: "For developers", icon: <CodeIcon /> },
];

function AgentReply({ msg }) {
  return (
    <div className={`agent ${msg.clarified ? "is-clarify" : ""}`}>
      <span className="agent-avatar" aria-hidden="true">
        <CloudMark />
      </span>
      <div className="agent-bubble">
        <div className="agent-body">
          <ReactMarkdown>{msg.text}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="agent" aria-label="Assistant is typing">
      <span className="agent-avatar" aria-hidden="true">
        <CloudMark />
      </span>
      <div className="agent-bubble typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setMode(d.mode))
      .catch(() => setMode("offline"));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || busy) return;
    // serverless functions are stateless: the browser carries the conversation
    const history = messages.map((m) => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.text,
    }));
    setInput("");
    setError(null);
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
      });
      if (!res.ok) throw new Error(`server returned ${res.status}`);
      const data = await res.json();
      // category/confidence/sources still come back for logs, but we don't
      // surface retrieval internals in the UI — only the answer and its tone.
      setMessages((m) => [
        ...m,
        { role: "agent", text: data.answer, clarified: data.clarified },
      ]);
    } catch {
      setError("We couldn't reach support just now. Give it a moment and try again.");
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  const empty = messages.length === 0;

  return (
    <div className="shell">
      <header className="topbar">
        <span className="wordmark">
          <span className="wordmark-icon">
            <CloudMark />
          </span>
          CloudNest
          <span className="wordmark-sub">Support</span>
        </span>
        {mode && (
          <span className={`status-pill ${mode === "claude" ? "is-online" : ""}`}>
            <span className="status-dot" />
            {mode === "claude" ? "Online" : "Limited mode"}
          </span>
        )}
      </header>

      <main className={`thread ${empty ? "is-empty" : ""}`}>
        {empty ? (
          <div className="hero">
            <span className="hero-badge">
              <CloudMark />
            </span>
            <h1 className="hero-title">How can we help today?</h1>
            <p className="hero-sub">
              Plans, billing, setup, syncing — ask anything and you'll get a straight answer.
            </p>
            <div className="cards">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  type="button"
                  className="card"
                  onClick={() => send(ex.label)}
                >
                  <span className="card-icon">{ex.icon}</span>
                  <span className="card-text">
                    <span className="card-label">{ex.label}</span>
                    <span className="card-hint">{ex.hint}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) =>
            msg.role === "user" ? (
              <div key={i} className="user">
                {msg.text}
              </div>
            ) : (
              <AgentReply key={i} msg={msg} />
            )
          )
        )}

        {busy && <TypingDots />}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask anything…"
          aria-label="Ask a question"
          autoFocus
        />
        <button type="submit" disabled={!input.trim() || busy} aria-label="Send">
          <SendIcon />
        </button>
      </form>
    </div>
  );
}
