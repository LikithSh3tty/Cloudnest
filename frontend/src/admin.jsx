import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { clearSession, loadSession } from "./session";
import "./admin.css";

function when(value) {
  if (!value) return null;
  const d = new Date(value);
  return isNaN(d) ? value : d.toLocaleString();
}

function Ticket({ ticket, open, onToggle }) {
  return (
    <div className={`admin-ticket${open ? " open" : ""}`}>
      <button className="admin-ticket-head" onClick={onToggle}>
        <span className="admin-when">{when(ticket.created_at)}</span>
        <span className="admin-user">
          {ticket.username ? (
            <>
              <span className="admin-avatar" aria-hidden="true">
                {ticket.username.slice(0, 1).toUpperCase()}
              </span>
              {ticket.username}
            </>
          ) : (
            <span className="admin-anon">not signed in</span>
          )}
        </span>
        <span className="admin-cat">{ticket.category}</span>
        <span className="admin-reason">{ticket.reason}</span>
        <span className="admin-conf">{ticket.confidence.toFixed(2)}</span>
        <span className="admin-q">{ticket.question}</span>
      </button>
      {open && (
        <div className="admin-detail">
          <dl className="admin-meta">
            <div>
              <dt>Raised by</dt>
              <dd>{ticket.username || "not signed in"}</dd>
            </div>
            <div>
              <dt>Signed in</dt>
              <dd>{when(ticket.signed_in_at) || "—"}</dd>
            </div>
            <div>
              <dt>Ticket</dt>
              <dd className="admin-mono">{ticket.id}</dd>
            </div>
          </dl>
          <div className="admin-convo">
            {ticket.conversation.map((m, i) => (
              <div key={i} className={`admin-msg admin-msg-${m.role}`}>
                <div className="admin-role">
                  {m.role === "user" ? ticket.username || "user" : "CloudNest"}
                </div>
                <div className="admin-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Admin() {
  const [session] = useState(() => loadSession());
  const [tickets, setTickets] = useState([]);
  const [error, setError] = useState("");
  const [openId, setOpenId] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // There is no login form here - sign-in happens once, on the front page.
    if (!session?.isAdmin) {
      window.location.replace("/");
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/tickets", {
          headers: { "X-Admin-User": session.username, "X-Admin-Token": session.token },
        });
        if (cancelled) return;
        if (res.status === 401) {
          clearSession();
          window.location.replace("/");
          return;
        }
        if (res.status === 503) {
          setError("Admin isn't configured on this deployment.");
          return;
        }
        if (!res.ok) {
          setError("Something went wrong loading tickets.");
          return;
        }
        const data = await res.json();
        setTickets(data.tickets || []);
      } catch {
        if (!cancelled) setError("Could not reach the server.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session]);

  function signOut() {
    clearSession();
    window.location.replace("/");
  }

  if (!session?.isAdmin) return null; // redirecting

  return (
    <div className="admin-wrap">
      <header className="admin-header">
        <div>
          <span className="admin-eyebrow">CloudNest admin</span>
          <h1>Escalated tickets</h1>
        </div>
        <span className="admin-count">{tickets.length}</span>
        <a className="admin-link" href="/">
          Support desk
        </a>
        <button type="button" className="admin-signout" onClick={signOut}>
          Sign out
        </button>
      </header>
      {error && <div className="admin-empty">{error}</div>}
      {!error && loading && <div className="admin-empty">Loading…</div>}
      {!error && !loading && tickets.length === 0 && (
        <div className="admin-empty">No tickets yet.</div>
      )}
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
