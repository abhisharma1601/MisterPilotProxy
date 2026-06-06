// Reads extension/.env and writes src/config.generated.ts
// Run automatically before every build via "prebuild:ext"

const fs = require('fs');
const path = require('path');

const envPath = path.join(__dirname, '..', '.env');
const outPath = path.join(__dirname, '..', 'src', 'config.generated.ts');

let backendUrl = 'http://localhost:8000';

if (fs.existsSync(envPath)) {
  const content = fs.readFileSync(envPath, 'utf8');
  const match = content.match(/^BACKEND_URL=(.+)$/m);
  if (match) {
    backendUrl = match[1].trim();
  } else {
    console.warn('[gen-config] BACKEND_URL not found in .env — using default');
  }
} else {
  console.warn('[gen-config] .env not found — using default');
}

fs.writeFileSync(
  outPath,
  `// Auto-generated from .env — do not edit manually\nexport const BACKEND_URL = '${backendUrl}';\n`
);

console.log(`[gen-config] BACKEND_URL → ${backendUrl}`);
