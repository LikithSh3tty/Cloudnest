// One sign-in for both roles. Nothing here is real authentication for an
// ordinary user: the browser asserts a name and no account exists. Only the
// admin half is checked, and it is checked by the server - we try the admin
// endpoint with whatever was typed, and let the 200 or 401 decide. That keeps
// the admin username out of this bundle entirely.
//
// Stored in localStorage, so you stay signed in across tabs and restarts until
// you sign out. That also means the admin password sits on disk for the admin
// session, which is the trade for not retyping it every visit.
const KEY = "cloudnest.session";

export function loadSession() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function clearSession() {
  localStorage.removeItem(KEY);
}

export async function signIn(username, password) {
  let isAdmin = false;
  try {
    const res = await fetch("/api/tickets", {
      headers: { "X-Admin-User": username, "X-Admin-Token": password },
    });
    isAdmin = res.ok;
  } catch {
    isAdmin = false; // unreachable server: treat as an ordinary visitor
  }
  const session = {
    username,
    signedInAt: new Date().toISOString(),
    isAdmin,
    // the password is kept only when it actually unlocks something
    token: isAdmin ? password : null,
  };
  localStorage.setItem(KEY, JSON.stringify(session));
  return session;
}
