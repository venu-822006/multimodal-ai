import { build } from 'esbuild';
import { createServer } from 'node:http';
import { copyFile, mkdir, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(root, 'dist');
const assetsDir = path.join(outDir, 'assets');
const mode = process.argv[2] || 'build';
const port = Number(process.env.PORT || (mode === 'preview' ? 4173 : 3000));
const apiTarget = process.env.FLOODSIGHT_API_PROXY || 'http://127.0.0.1:8000';

async function compile() {
  await mkdir(assetsDir, { recursive: true });
  await build({
    entryPoints: [path.join(root, 'src/main.jsx')],
    bundle: true,
    minify: mode !== 'dev',
    sourcemap: mode === 'dev',
    platform: 'browser',
    format: 'iife',
    target: ['es2020'],
    outfile: path.join(assetsDir, 'app.js'),
    assetNames: 'media/[name]-[hash]',
    loader: { '.png': 'file', '.jpg': 'file', '.jpeg': 'file', '.svg': 'file', '.woff': 'file', '.woff2': 'file' },
    logLevel: 'info'
  });
  await copyFile(path.join(root, 'index.html'), path.join(outDir, 'index.html'));
}

function contentType(file) {
  const ext = path.extname(file).toLowerCase();
  return ({ '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json; charset=utf-8', '.svg':'image/svg+xml', '.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.woff2':'font/woff2' })[ext] || 'application/octet-stream';
}

// Proxy /api (and /health) to the FastAPI backend so the browser can use
// same-origin requests and window.FLOODSIGHT_API_BASE_URL can stay empty.
async function proxy(req, res, pathname, search) {
  const target = `${apiTarget}${pathname}${search}`;
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = chunks.length ? Buffer.concat(chunks) : undefined;
  const headers = { accept: req.headers.accept || 'application/json' };
  // Only forward a body content type on methods that actually carry a body.
  if (body && req.headers['content-type']) headers['content-type'] = req.headers['content-type'];
  if (req.headers.authorization) headers.authorization = req.headers.authorization;
  // Session auth: carry the HttpOnly cookie and the declared origin through the proxy
  // so server-side sessions and origin checks behave exactly as if the API were same-origin.
  if (req.headers.cookie) headers.cookie = req.headers.cookie;
  if (req.headers.origin) headers.origin = req.headers.origin;
  if (req.headers.referer) headers.referer = req.headers.referer;
  try {
    const upstream = await fetch(target, { method: req.method, headers, body, redirect: 'manual' });
    const payload = Buffer.from(await upstream.arrayBuffer());
    const out = { 'Cache-Control': 'no-store' };
    for (const key of ['content-type', 'content-disposition', 'location', 'retry-after']) {
      const value = upstream.headers.get(key);
      if (value) out[key] = value;
    }
    // Set-Cookie must survive the proxy or the browser never receives the session cookie.
    const setCookies = typeof upstream.headers.getSetCookie === 'function' ? upstream.headers.getSetCookie() : [];
    if (setCookies.length) out['set-cookie'] = setCookies;
    res.writeHead(upstream.status, out);
    res.end(payload);
  } catch {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { code: 'backend_unreachable', message: 'The analysis backend is not responding.' } }));
  }
}

async function serve() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      const pathname = decodeURIComponent(url.pathname);
      if (pathname.startsWith('/api/') || pathname === '/api' || pathname.startsWith('/health')) {
        return proxy(req, res, pathname, url.search);
      }
      let target = path.resolve(outDir, `.${pathname}`);
      if (target !== outDir && !target.startsWith(outDir + path.sep)) {
        res.writeHead(403); res.end('Forbidden'); return;
      }
      if (!existsSync(target) || (await stat(target)).isDirectory()) target = path.join(outDir, 'index.html');
      const body = await readFile(target);
      res.writeHead(200, { 'Content-Type': contentType(target), 'Cache-Control': 'no-cache' });
      res.end(body);
    } catch {
      res.writeHead(500); res.end('Unable to serve this resource.');
    }
  });
  server.listen(port, '0.0.0.0', () => console.log(`FloodSight AI ${mode} server running at http://localhost:${port}`));
  if (mode !== 'preview') console.log(`Proxying /api and /health to ${apiTarget}`);
}

if (mode === 'dev') {
  await compile();
  await serve();
} else if (mode === 'build') {
  await compile();
  console.log('Production files written to dist/');
} else if (mode === 'preview') {
  if (!existsSync(path.join(outDir, 'index.html'))) {
    console.error('No production build found. Run "npm run build" first.');
    process.exit(1);
  }
  await serve();
} else {
  console.error(`Unknown command: ${mode}`);
  process.exit(1);
}
