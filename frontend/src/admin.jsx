import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./admin.css";

function Login({ onSubmit, error }) {
  const [pw, setPw] = useState("");
  return (
    <form
      className="admin-login"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(pw);
      }}
    >
      <h1>CloudNest Admin</h1>
      <p>Enter the admin password to view escalated tickets.</p>
      <input
        type="password"
        value={pw}
        onChange={(e) => setPw(e.target.value)}
        placeholder="Password"
        autoFocus
      />
      <button type="submit">Sign in</button>
      {error && <div className="admin-error">{error}</div>}
    </form>
  );
}

function Ticket({ ticket, open, onToggle }) {
  const when = new Date(ticket.created_at).toLocaleString();
  return (
    <div className={`admin-ticket${open ? " open" : ""}`}>
      <button className="admin-ticket-head" onClick={onToggle}>
        <span className="admin-when">{when}</span>
        <span className="admin-cat">{ticket.category}</span>
        <span className="admin-reason">{ticket.reason}</span>
        <span className="admin-conf">{ticket.confidence.toFixed(2)}</span>
        <span className="admin-q">{ticket.question}</span>
      </button>
      {open && (
        <div className="admin-convo">
          {ticket.conversation.map((m, i) => (
            <div key={i} className={`admin-msg admin-msg-${m.role}`}>
              <div className="admin-role">{m.role}</div>
              <div className="admin-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Admin() {
  const [authed, setAuthed] = useState(false);
  const [tickets, setTickets] = useState([]);
  const [error, setError] = useState("");
  const [openId, setOpenId] = useState(null);
  const [loading, setLoading] = useState(false);

  async function load(token) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/tickets", { headers: { "X-Admin-Token": token } });
      if (res.status === 401) {
        sessionStorage.removeItem("adminToken");
        setAuthed(false);
        setError("Wrong password.");
        return;
      }
      if (res.status === 503) {
        setAuthed(false);
        setError("Admin isn't configured yet.");
        return;
      }
      if (!res.ok) {
        setAuthed(false);
        setError("Something went wrong.");
        return;
      }
      const data = await res.json();
      sessionStorage.setItem("adminToken", token);
      setTickets(data.tickets || []);
      setAuthed(true);
    } catch (e) {
      setAuthed(false);
      setError("Could not reach the server.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const saved = sessionStorage.getItem("adminToken");
    if (saved) load(saved);
  }, []);

  if (!authed) return <Login onSubmit={load} error={error} />;

  return (
    <div className="admin-wrap">
      <header className="admin-header">
        <h1>Escalated tickets</h1>
        <span className="admin-count">{tickets.length}</span>
      </header>
      {loading && <div className="admin-empty">Loading…</div>}
      {!loading && tickets.length === 0 && <div className="admin-empty">No tickets yet.</div>}
      {tickets.map((t) => (
        <Ticket
          key={t.id}
          ticket={t}
          open={openId === t.id}
          onToggle={() => setOpenId(openId === t.id ? null : t.id)}
        />
      ))}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Admin />
  </React.StrictMode>
);
