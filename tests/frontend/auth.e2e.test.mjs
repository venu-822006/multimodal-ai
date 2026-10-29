import { test } from 'node:test';
import assert from 'node:assert/strict';

// End-to-end HTTP check of the real backend (and the dev proxy when pointed at it).
// Run with: FLOODSIGHT_E2E_BASE_URL=http://localhost:3000 node --test tests/frontend/
const BASE = process.env.FLOODSIGHT_E2E_BASE_URL;
const skip = BASE ? false : 'set FLOODSIGHT_E2E_BASE_URL to run the live authentication check';

const COOKIE_NAME = 'floodsight_session';

function createJar() {
  const jar = new Map();
  return {
    header() {
      return [...jar.entries()].map(([name, value]) => `${name}=${value}`).join('; ');
    },
    store(response) {
      const setCookies = typeof response.headers.getSetCookie === 'function' ? response.headers.getSetCookie() : [];
      for (const raw of setCookies) {
        const pair = raw.split(';')[0];
        const index = pair.indexOf('=');
        const name = pair.slice(0, index).trim();
        const value = pair.slice(index + 1).trim();
        if (!value) jar.delete(name);
        else jar.set(name, value);
      }
    },
  };
}

async function call(path, { method = 'GET', body, jar } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (BASE.startsWith('http://localhost:3000') || BASE.startsWith('http://127.0.0.1:3000')) {
    headers.Origin = BASE;
  }
  const cookie = jar ? jar.header() : '';
  if (cookie) headers.Cookie = cookie;
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    redirect: 'manual',
  });
  if (jar) jar.store(response);
  return response;
}

const email = `e2e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}@example.org`;
const password = 'correct-horse-42';

test('register -> me -> logout -> denied', { skip }, async () => {
  const jar = createJar();

  const registered = await call('/api/auth/register', {
    method: 'POST',
    jar,
    body: { email, password, full_name: 'E2E Check' },
  });
  const created = await registered.json();
  assert.equal(registered.status, 201, JSON.stringify(created));
  assert.equal(created.user.email, email);
  assert.ok(!('password_hash' in created.user));

  const meBefore = await call('/api/auth/me', { jar });
  assert.equal(meBefore.status, 200);
  assert.equal((await meBefore.json()).email, email);

  const loggedOut = await call('/api/auth/logout', { method: 'POST', jar });
  assert.equal(loggedOut.status, 204);

  const meAfter = await call('/api/auth/me', { jar });
  assert.equal(meAfter.status, 401);

  const wrongPassword = await call('/api/auth/login', {
    method: 'POST',
    jar: createJar(),
    body: { email, password: 'definitely-wrong' },
  });
  assert.equal(wrongPassword.status, 401);

  const relogin = await call('/api/auth/login', { method: 'POST', jar: createJar(), body: { email, password } });
  assert.equal(relogin.status, 200);
});

test('invalid cookie is rejected', { skip }, async () => {
  const response = await fetch(`${BASE}/api/auth/me`, {
    headers: { Cookie: `${COOKIE_NAME}=invalid-token-value` },
    redirect: 'manual',
  });
  assert.equal(response.status, 401);
});

test('protected endpoint rejects unauthenticated requests', { skip }, async () => {
  const response = await fetch(`${BASE}/api/auth/me`, { redirect: 'manual' });
  assert.equal(response.status, 401);
});
