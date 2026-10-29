import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const bundlePath = new URL('../../dist/assets/app.js', import.meta.url);
const available = existsSync(bundlePath);

test(
  'production bundle wires authentication to the backend',
  { skip: available ? false : 'run "npm run build" first' },
  async () => {
    const source = await readFile(bundlePath, 'utf8');
    assert.ok(source.includes('/api/auth/login'), 'login must call the backend');
    assert.ok(source.includes('/api/auth/register'), 'signup must call the backend');
    assert.ok(source.includes('/api/auth/me'), 'session restore must call the backend');
    assert.ok(source.includes('/api/auth/logout'), 'logout must call the backend');
    assert.ok(source.includes('floodsight:unauthorized'), '401 responses must reset auth state');
    assert.ok(source.includes('VERIFYING SESSION'), 'the UI must wait for backend confirmation');
    assert.ok(!source.includes('floodsight-access'), 'sessionStorage must not prove authentication');
  },
);
