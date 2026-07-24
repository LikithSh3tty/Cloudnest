import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function ChatIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 5h14a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H9l-4 3V6a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function BookIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 6h6a1 1 0 0 1 1 1v11H6a1 1 0 0 1-1-1V6ZM19 6h-6a1 1 0 0 0-1 1v11h6a1 1 0 0 0 1-1V6Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 6v12M6 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 19V6M6 11l6-6 6 6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 12h13M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const GUIDANCE = [
  { label: "Free up storage", ask: "How do I free up storage space on CloudNest?" },
  { label: "Resolve sync issues", ask: "My files aren't syncing on my laptop — how do I fix it?" },
  { label: "Review plan limits", ask: "What are the storage and device limits on each plan?" },
];

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function AgentReply({ msg }) {
  return (
    <div className="agent">
      <span className="agent-avatar" aria-hidden="true">
        C
      </span>
      <div className="agent-main">
        <div className="agent-head">
          <span className="agent-name">CloudNest Support</span>
          <span className="agent-time">JUST NOW</span>
        </div>
        <div className="agent-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function TypingReply() {
  return (
    <div className="agent">
      <span className="agent-avatar" aria-hidden="true">
        C
      </span>
      <div className="agent-main">
        <div className="agent-head">
          <span className="agent-name">CloudNest Support</span>
          <span className="agent-time">TYPING</span>
        </div>
        <div className="typing" aria-label="Assistant is typing">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
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
  const [deskId] = useState(() => String(Math.floor(Math.random() * 900) + 100));
  const scrollRef = useRef(null);
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
      setMessages((m) => [...m, { role: "agent", text: data.answer }]);
    } catch {
      setError("We couldn't reach support just now. Give it a moment and try again.");
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function newRequest() {
    setMessages([]);
    setInput("");
    setError(null);
    inputRef.current?.focus();
  }

  const online = mode === "claude";
  const empty = messages.length === 0 && !busy && !error;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">C</span>
          <span className="brand-text">
            CloudNest
            <span className="brand-sub">Support desk</span>
          </span>
        </div>

        <nav className="nav">
          <span className="nav-label">Workspace</span>
          <button type="button" className="nav-item is-active">
            <ChatIcon />
            Conversation
          </button>
          <button type="button" className="nav-item" onClick={() => send("What can you help me with?")}>
            <BookIcon />
            Help centre
          </button>
          <button type="button" className="nav-item" onClick={newRequest}>
            <PlusIcon />
            New request
          </button>
        </nav>

        <p className="sidebar-foot">
          A quieter support experience, designed to keep the answer — not the interface — at
          the centre.
        </p>
      </aside>

      <main className="main">
        <div className="main-bar">
          <div>
            <span className="eyebrow">Support desk / {deskId}</span>
            <h2 className="main-title">CloudNest account support</h2>
          </div>
          {mode && (
            <span className={`avail ${online ? "is-online" : ""}`}>
              <span className="avail-dot" />
              {online ? "Support available" : "Limited support"}
            </span>
          )}
        </div>

        <div className="main-scroll" ref={scrollRef}>
          <header className="hero">
            <span className="eyebrow">{greeting()}</span>
            <h1 className="hero-title">
              Clear answers for <em>your cloud.</em>
            </h1>
            <p className="hero-sub">
              Start with what happened. We'll keep the conversation focused and help you find
              the next right step.
            </p>
          </header>

          {!empty && (
            <div className="conversation">
              {messages.map((msg, i) =>
                msg.role === "user" ? (
                  <div key={i} className="user">
                    {msg.text}
                  </div>
                ) : (
                  <AgentReply key={i} msg={msg} />
                )
              )}
              {busy && <TypingReply />}
              {error && <div className="error">{error}</div>}
              <div ref={endRef} />
            </div>
          )}
        </div>

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
            placeholder="Reply to support…"
            aria-label="Reply to support"
            autoFocus
          />
          <button type="submit" disabled={!input.trim() || busy} aria-label="Send">
            <SendIcon />
          </button>
        </form>
      </main>

      <aside className="context">
        <div className="ctx-block">
          <span className="eyebrow">This conversation</span>
          <p className="ctx-value">CloudNest support</p>
          <p className="ctx-note">
            Answers stay focused on what you asked. Account details appear here when they add
            useful context.
          </p>
        </div>

        <div className="ctx-divider" />

        <div className="ctx-block">
          <span className="eyebrow">Related guidance</span>
          <div className="ctx-links">
            {GUIDANCE.map((g) => (
              <button key={g.label} type="button" className="ctx-link" onClick={() => send(g.ask)}>
                {g.label}
                <ArrowIcon />
              </button>
            ))}
          </div>
        </div>

        <div className="ctx-card">
          <p className="ctx-card-title">Designed for calm</p>
          <p className="ctx-card-text">
            No gradients, mascot language, fake metrics, or overloaded cards. The interface
            earns its premium feel through restraint and careful hierarchy.
          </p>
        </div>
      </aside>
    </div>
  );
}
