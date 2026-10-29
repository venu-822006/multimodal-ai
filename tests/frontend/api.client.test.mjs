import { test } from 'node:test';
import assert from 'node:assert/strict';

// src/api.js reads window.FLOODSIGHT_API_BASE_URL at import time and dispatches
// auth events on window, so a minimal window is installed before it is imported.
const events = [];
globalThis.window = {
  FLOODSIGHT_API_BASE_URL: '',
  dispatchEvent: (event) => { events.push(event); return true; },
};

const { api } = await import('../../src/api.js');

function mockFetch(handler) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return handler(url, options);
  };
  return calls;
}

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => null },
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

const USER = { id: '11111111-1111-1111-1111-111111111111', email: 'analyst@example.org', full_name: 'Analyst', is_active: true };

test('login posts credentials to the real backend endpoint', async () => {
  const calls = mockFetch(() => jsonResponse(200, { user: USER }));
  const result = await api.login({ email: 'analyst@example.org', password: 'correct-horse-42', remember: true });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/auth/login');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.credentials, 'include');
  assert.equal(calls[0].options.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[0].options.body), { email: 'analyst@example.org', password: 'correct-horse-42', remember: true });
  assert.equal(result.user.email, 'analyst@example.org');
});

test('register posts to the real backend endpoint', async () => {
  const calls = mockFetch(() => jsonResponse(201, { user: USER }));
  await api.register({ email: 'analyst@example.org', password: 'correct-horse-42', full_name: 'Analyst' });
  assert.equal(calls[0].url, '/api/auth/register');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.credentials, 'include');
});

test('me fetches the current user with credentials', async () => {
  const calls = mockFetch(() => jsonResponse(200, USER));
  const user = await api.me();
  assert.equal(calls[0].url, '/api/auth/me');
  assert.equal(calls[0].options.credentials, 'include');
  assert.equal(user.email, 'analyst@example.org');
});

test('logout posts to the backend and tolerates 204', async () => {
  const calls = mockFetch(() => jsonResponse(204, null));
  const result = await api.logout();
  assert.equal(calls[0].url, '/api/auth/logout');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(result, null);
});

test('invalid credentials reject with backend message and status', async () => {
  mockFetch(() => jsonResponse(401, { error: { code: 'invalid_credentials', message: 'Invalid email or password.' } }));
  await assert.rejects(
    () => api.login({ email: 'analyst@example.org', password: 'wrong' }),
    (error) => {
      assert.equal(error.status, 401);
      assert.equal(error.message, 'Invalid email or password.');
      return true;
    },
  );
});

test('a 401 dispatches a global unauthorized event', async () => {
  events.length = 0;
  mockFetch(() => jsonResponse(401, { error: { code: 'unauthenticated', message: 'Authentication is required for this endpoint.' } }));
  await assert.rejects(() => api.me());
  assert.equal(events.length, 1);
  assert.equal(events[0].type, 'floodsight:unauthorized');
});

test('duplicate registration surfaces the conflict message', async () => {
  mockFetch(() => jsonResponse(409, { error: { code: 'conflict', message: 'An account with this email already exists.' } }));
  await assert.rejects(
    () => api.register({ email: 'analyst@example.org', password: 'correct-horse-42' }),
    (error) => {
      assert.equal(error.status, 409);
      assert.equal(error.message, 'An account with this email already exists.');
      return true;
    },
  );
});

test('validation errors surface field details without exposing internals', async () => {
  mockFetch(() => jsonResponse(422, { error: { code: 'validation_error', message: 'Request payload failed validation.', details: { fields: [{ field: 'email', message: 'Enter a valid email address.' }] } } }));
  await assert.rejects(
    () => api.register({ email: 'nope', password: 'correct-horse-42' }),
    (error) => {
      assert.equal(error.status, 422);
      assert.match(error.message, /Enter a valid email address\./);
      return true;
    },
  );
});
