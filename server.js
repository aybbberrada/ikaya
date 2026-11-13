// Zero-dependency HTTPS static server for the Ikaya gaze tracker.
// iOS Safari only grants camera access on secure origins, so we serve over HTTPS
// with a self-signed certificate generated on first run (requires `openssl`).
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const PORT = Number(process.env.PORT) || 8443;
const HTTP_PORT = Number(process.env.HTTP_PORT) || 8080;
const ROOT = path.join(__dirname, 'public');
const CERT_DIR = path.join(__dirname, 'certs');
const KEY = path.join(CERT_DIR, 'key.pem');
const CERT = path.join(CERT_DIR, 'cert.pem');

function lanAddresses() {
  const out = [];
  for (const list of Object.values(os.networkInterfaces())) {
    for (const a of list || []) {
      if (a.family === 'IPv4' && !a.internal) out.push(a.address);
    }
  }
  return out;
}

function ensureCert() {
  if (fs.existsSync(KEY) && fs.existsSync(CERT)) return;
  fs.mkdirSync(CERT_DIR, { recursive: true });
  const ips = lanAddresses();
  const san = ['DNS:localhost', 'IP:127.0.0.1', ...ips.map((ip) => `IP:${ip}`)].join(',');
  console.log('Generating self-signed certificate…');
  try {
    execSync(
      `openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 825 ` +
        `-keyout "${KEY}" -out "${CERT}" -subj "/CN=ikaya.local" ` +
        `-addext "subjectAltName=${san}"`,
      { stdio: 'ignore' }
    );
  } catch {
    // Older OpenSSL / LibreSSL without -addext
    execSync(
      `openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 825 ` +
        `-keyout "${KEY}" -out "${CERT}" -subj "/CN=ikaya.local"`,
      { stdio: 'ignore' }
    );
  }
}

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

function handler(req, res) {
  let urlPath = decodeURIComponent(new URL(req.url, 'https://x').pathname);
  if (urlPath.endsWith('/')) urlPath += 'index.html';
  const file = path.normalize(path.join(ROOT, urlPath));
  if (!file.startsWith(ROOT)) {
    res.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404).end('Not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  });
}

try {
  ensureCert();
} catch (e) {
  console.error('Could not create a certificate. Is `openssl` installed?\n', e.message);
  process.exit(1);
}

https
  .createServer({ key: fs.readFileSync(KEY), cert: fs.readFileSync(CERT) }, handler)
  .listen(PORT, '0.0.0.0', () => {
    console.log('\nIkaya gaze tracker is running.\n');
    console.log(`  On this computer:  https://localhost:${PORT}`);
    for (const ip of lanAddresses()) console.log(`  On your iPhone:    https://${ip}:${PORT}`);
    console.log('\nThe iPhone must be on the same Wi-Fi. Safari will warn about the');
    console.log('self-signed certificate: tap "Show Details" → "visit this website".\n');
  });

// Plain HTTP on localhost is also a secure context for desktop testing.
http.createServer(handler).listen(HTTP_PORT, '127.0.0.1');
