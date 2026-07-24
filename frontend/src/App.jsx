import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "How much does the Pro plan cost?",
  "Which plans include API access?",
  "My files aren't syncing on my laptop",
];

function CloudMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 12h13M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ConfidenceMeter({ value }) {
  return (
    <span className="meter" role="img" aria-label={`confidence ${value.toFixed(2)}`}>
      <span className="meter-fill" style={{ width: `${Math.min(value, 1) * 100}%` }} />
    </span>
  );
}

function AgentReply({ msg }) {
  const isClarify = msg.category === "clarify-request";
  return (
    <div className={`agent ${isClarify ? "agent-clarify" : ""}`}>
      <div className="agent-meta">
        <span className="route">{msg.category}</span>
        <span className="conf">confidence {msg.confidence.toFixed(2)}</span>
        <ConfidenceMeter value={msg.confidence} />
      </div>
      <div className="agent-body">{msg.text}</div>
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
      const isClarify = data.clarified; // the graph reports which branch it took
      setMessages((m) => [
        ...m,
        {
          role: "agent",
          text: data.answer,
          category: isClarify ? "clarify-request" : data.category,
          confidence: data.confidence,
        },
      ]);
    } catch {
      setError("Can't reach the support service. Check that the server is running on port 8000.");
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <span className="wordmark">
          <CloudMark />
          CloudNest Support
        </span>
        {mode && <span className={`mode mode-${mode}`}>{mode === "claude" ? "assistant online" : mode}</span>}
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about plans, billing, setup, or troubleshooting.</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} type="button" onClick={() => send(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="user">
              {msg.text}
            </div>
          ) : (
            <AgentReply key={i} msg={msg} />
          )
        )}

        {busy && <div className="status">searching cloudnest_docs…</div>}
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
          placeholder="Ask a question"
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
