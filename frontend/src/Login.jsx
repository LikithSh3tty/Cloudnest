import React, { useState } from "react";
import { signIn } from "./session";

export default function Login({ onSignedIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (busy || !username.trim() || !password) return;
    setBusy(true);
    const session = await signIn(username.trim(), password);
    setBusy(false);
    // Admins get the ticket queue, everyone else gets the support desk.
    if (session.isAdmin) window.location.href = "/admin";
    else onSignedIn(session);
  }

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={submit}>
        <div className="brand login-brand">
          <span className="brand-mark">C</span>
          <span className="brand-text">
            CloudNest
            <span className="brand-sub">Support desk</span>
          </span>
        </div>

        <span className="eyebrow">Sign in</span>
        <h1 className="login-title">Welcome back</h1>
        <p className="login-note">
          Sign in to open a support conversation. Admins land on the ticket queue instead.
        </p>

        <label className="login-field">
          <span>Username</span>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Your name"
            autoComplete="username"
            autoFocus
          />
        </label>

        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            autoComplete="current-password"
          />
        </label>

        <button className="login-submit" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>

        <p className="login-foot">
          Demo sign-in: any username and password opens the support desk.
        </p>
      </form>
    </div>
  );
}
