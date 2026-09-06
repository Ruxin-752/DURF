// Integration check against Miniflare's actual local D1 engine, with no study rows.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const ts = require(path.join(root, 'web/node_modules/typescript'));
const { Miniflare } = require(path.join(root, 'web/node_modules/miniflare'));
require.extensions['.ts'] = (module, file) => module._compile(ts.transpileModule(fs.readFileSync(file, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }, fileName: file,
}).outputText, file);
const { handleSyntheticDiagnosticRequest } = require(path.join(root, 'web/lib/research-diagnostic.ts'));

(async () => {
  const runtime = new Miniflare({ modules: true, compatibilityDate: '2026-05-15',
    script: 'export default { fetch() { return new Response("synthetic-only-local-check"); } }',
    d1Databases: ['DB'], port: 0,
  });
  try {
    const DB = await runtime.getD1Database('DB');
    const token = 'local-synthetic-test-credential-only';
    const origin = 'https://synthetic.local';
    const response = await handleSyntheticDiagnosticRequest(new Request(origin + '/api/research/diagnostic', {
      method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', Authorization: 'Bearer ' + token },
      body: '{}',
    }), { DB, ADMIN_EXPORT_TOKEN: token, SITE_ORIGIN: origin });
    const body = await response.json();
    const remaining = await DB.prepare('SELECT COUNT(*) AS count FROM research_diagnostics').first('count');
    const tables = await DB.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions','events','feedback')").all();
    const result = { scope: 'local Miniflare D1 engine, not production', status: response.status,
      response: body, remaining_diagnostic_rows: remaining, study_tables_created: tables.results.length };
    const target = path.join(root, 'artifacts/synthetic-acceptance-20260906/local-d1-roundtrip.json');
    fs.writeFileSync(target, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
    if (response.status !== 200 || body.ok !== true || remaining !== 0 || tables.results.length !== 0) {
      throw new Error('Local D1 roundtrip failed');
    }
  } finally { await runtime.dispose(); }
})().catch((error) => { console.error(error.message); process.exitCode = 1; });
