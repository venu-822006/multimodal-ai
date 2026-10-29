// Set window.FLOODSIGHT_API_BASE_URL to the FastAPI origin when the API runs separately.
// Empty base uses same-origin /api routes and only works when a backend proxy is configured.
const API_BASE = (window.FLOODSIGHT_API_BASE_URL || '').replace(/\/$/, '');

async function request(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const response = await fetch(`${API_BASE}${path}`, {
    // The session lives in an HttpOnly cookie, so credentials must be sent.
    credentials: 'include',
    ...rest,
    headers: { 'Content-Type': 'application/json', ...(extraHeaders || {}) }
  });
  if (!response.ok) {
    const fallback = `Request failed (${response.status})`;
    let message = fallback;
    try {
      const body = await response.json();
      const error = body && body.error;
      const detail = body && body.detail;
      if (error && typeof error === 'object') {
        message = typeof error.message === 'string' ? error.message : fallback;
        const fields = error.details && error.details.fields;
        if (Array.isArray(fields) && fields[0] && fields[0].message) {
          message = `${message} ${fields[0].message}`;
        }
      } else if (typeof error === 'string') {
        message = error;
      } else if (typeof detail === 'string') {
        message = detail;
      } else if (Array.isArray(detail) && detail[0] && detail[0].msg) {
        message = detail[0].msg;
      } else if (body && typeof body.message === 'string') {
        message = body.message;
      }
    } catch {
      const text = await response.text().catch(() => '');
      if (text) message = text;
    }
    const failure = new Error(message);
    failure.status = response.status;
    // A 401 means the server-side session is missing, invalid, expired or revoked.
    // Let the app-wide auth state react instead of leaving stale "logged in" UI.
    if (response.status === 401) {
      window.dispatchEvent(new Event('floodsight:unauthorized'));
    }
    throw failure;
  }
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  register: (payload) => request('/api/auth/register', {
    method: 'POST', body: JSON.stringify(payload)
  }),
  login: (payload) => request('/api/auth/login', {
    method: 'POST', body: JSON.stringify(payload)
  }),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
  me: () => request('/api/auth/me'),
  listAnalyses: () => request('/api/analyses'),
  getAnalysis: (id) => request(`/api/analyses/${encodeURIComponent(id)}`),
  getStatus: (id) => request(`/api/analyses/${encodeURIComponent(id)}/status`),
  getFacts: (id) => request(`/api/analyses/${encodeURIComponent(id)}/facts`),
  getMap: (id) => request(`/api/analyses/${encodeURIComponent(id)}/map`),
  getReport: (id) => request(`/api/analyses/${encodeURIComponent(id)}/report`),
  createAnalysis: (payload) => request('/api/analyses', {
    method: 'POST', body: JSON.stringify(payload)
  })
};
