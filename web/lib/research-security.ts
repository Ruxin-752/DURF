import { initializeDatabase } from './db';
import {
  CONSENT_VERSION,
  type ResearchSessionGrantRequest,
  type SessionRecord,
} from './research-types';

export const RESEARCH_SESSION_COOKIE = '__Host-durf_research_session';
export const RESEARCH_SESSION_TTL_MS = 2 * 60 * 60 * 1000;

export interface ResearchSessionTokenRow {
  token_hash: string;
  session_id: string;
  anonymous_user_id: string;
  consent_version: string;
  consented_at: number;
  issued_at: number;
  expires_at: number;
  revoked_at: number | null;
}

export interface IssuedResearchSession {
  cookie: string;
  expiresAt: number;
}

export type ResearchSessionVerification =
  | { ok: true; tokenHash: string; expiresAt: number }
  | { ok: false; error: string };

export interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  resetAt: number;
}

const pepperPromises = new WeakMap<object, Promise<string>>();

function randomOpaqueValue(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/gu, '-').replace(/\//gu, '_').replace(/=+$/gu, '');
}

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (value) => value.toString(16).padStart(2, '0')).join('');
}

export async function sha256Hex(value: string): Promise<string> {
  return bytesToHex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)));
}

async function hmacSha256Hex(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  return bytesToHex(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(value)));
}

function cookieValue(request: Request, name: string): string | null {
  const header = request.headers.get('cookie');
  if (!header) return null;
  for (const part of header.split(';')) {
    const separator = part.indexOf('=');
    if (separator < 0) continue;
    if (part.slice(0, separator).trim() === name) return part.slice(separator + 1).trim();
  }
  return null;
}

export function buildResearchSessionCookie(
  token: string,
  expiresAt: number,
  now = Date.now(),
): string {
  return [
    `${RESEARCH_SESSION_COOKIE}=${token}`,
    'Path=/',
    'HttpOnly',
    'Secure',
    'SameSite=Strict',
    `Expires=${new Date(expiresAt).toUTCString()}`,
    `Max-Age=${Math.max(0, Math.floor((expiresAt - now) / 1000))}`,
  ].join('; ');
}

export function clearResearchSessionCookie(): string {
  return `${RESEARCH_SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`;
}

async function getOrCreateSecurityPepper(db: D1Database): Promise<string> {
  const cached = pepperPromises.get(db as object);
  if (cached) return cached;
  const promise = (async () => {
    await initializeDatabase(db);
    const generated = randomOpaqueValue();
    await db
      .prepare(
        `INSERT INTO research_security_secrets (secret_name, secret_value, created_at)
         VALUES ('network_hash_pepper', ?, ?)
         ON CONFLICT(secret_name) DO NOTHING`,
      )
      .bind(generated, Date.now())
      .run();
    const row = await db
      .prepare(
        `SELECT secret_value FROM research_security_secrets
         WHERE secret_name = 'network_hash_pepper'`,
      )
      .first<{ secret_value: string }>();
    if (!row?.secret_value) throw new Error('research security pepper unavailable');
    return row.secret_value;
  })();
  pepperPromises.set(db as object, promise);
  promise.catch(() => pepperPromises.delete(db as object));
  return promise;
}

/** Returns only a keyed hash. The raw network address is never persisted. */
export async function researchRateLimitKey(
  db: D1Database,
  request: Request,
  scope: string,
): Promise<string> {
  const rawNetworkAddress = request.headers.get('cf-connecting-ip')?.trim() || 'unavailable';
  const boundedAddress = rawNetworkAddress.slice(0, 128);
  const pepper = await getOrCreateSecurityPepper(db);
  return `${scope}:${await hmacSha256Hex(pepper, boundedAddress)}`;
}

export async function consumeResearchRateLimit(
  db: D1Database,
  bucketKey: string,
  limit: number,
  windowMs: number,
  now = Date.now(),
): Promise<RateLimitResult> {
  await initializeDatabase(db);
  const windowStart = Math.floor(now / windowMs) * windowMs;
  const resetAt = windowStart + windowMs;
  const row = await db
    .prepare(
      `INSERT INTO research_rate_limits (
         bucket_key, window_start, request_count, expires_at
       ) VALUES (?, ?, 1, ?)
       ON CONFLICT(bucket_key, window_start) DO UPDATE SET
         request_count = research_rate_limits.request_count + 1,
         expires_at = excluded.expires_at
       RETURNING request_count`,
    )
    .bind(bucketKey, windowStart, resetAt + windowMs)
    .first<{ request_count: number }>();
  if (!row || !Number.isSafeInteger(row.request_count)) {
    throw new Error('research rate limit update failed');
  }
  return {
    allowed: row.request_count <= limit,
    limit,
    remaining: Math.max(0, limit - row.request_count),
    resetAt,
  };
}

export async function purgeExpiredResearchSecurityRows(
  db: D1Database,
  now = Date.now(),
): Promise<void> {
  await initializeDatabase(db);
  await db.batch([
    db.prepare('DELETE FROM research_rate_limits WHERE expires_at < ?').bind(now),
    db.prepare('DELETE FROM research_session_tokens WHERE expires_at < ?').bind(now),
  ]);
}

export async function issueResearchSessionToken(
  db: D1Database,
  grant: ResearchSessionGrantRequest,
  now = Date.now(),
): Promise<IssuedResearchSession> {
  await initializeDatabase(db);
  const token = randomOpaqueValue();
  const tokenHash = await sha256Hex(token);
  const expiresAt = now + RESEARCH_SESSION_TTL_MS;
  await db
    .prepare(
      `INSERT INTO research_session_tokens (
         token_hash, session_id, anonymous_user_id, consent_version,
         consented_at, issued_at, expires_at, revoked_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
       ON CONFLICT(session_id) DO UPDATE SET
         token_hash = excluded.token_hash,
         anonymous_user_id = excluded.anonymous_user_id,
         consent_version = excluded.consent_version,
         consented_at = excluded.consented_at,
         issued_at = excluded.issued_at,
         expires_at = excluded.expires_at,
         revoked_at = NULL`,
    )
    .bind(
      tokenHash,
      grant.sessionId,
      grant.anonymousUserId,
      grant.consentVersion,
      grant.consentedAt,
      now,
      expiresAt,
    )
    .run();
  return { cookie: buildResearchSessionCookie(token, expiresAt, now), expiresAt };
}

export function validateStoredResearchSession(
  row: ResearchSessionTokenRow | null,
  expected: Pick<
    SessionRecord,
    'sessionId' | 'anonymousUserId' | 'consentVersion' | 'consentedAt'
  >,
  now = Date.now(),
): ResearchSessionVerification {
  if (!row) return { ok: false, error: 'Anonymous research session not found' };
  if (row.revoked_at !== null || row.expires_at <= now) {
    return { ok: false, error: 'Anonymous research session has expired' };
  }
  if (row.consent_version !== CONSENT_VERSION || expected.consentVersion !== CONSENT_VERSION) {
    return { ok: false, error: 'The consent form has changed. Please confirm it again.' };
  }
  if (
    row.session_id !== expected.sessionId ||
    row.anonymous_user_id !== expected.anonymousUserId ||
    row.consented_at !== expected.consentedAt
  ) {
    return { ok: false, error: 'Anonymous research session does not match the request' };
  }
  return { ok: true, tokenHash: row.token_hash, expiresAt: row.expires_at };
}

export async function verifyResearchSessionToken(
  db: D1Database,
  request: Request,
  expected: Pick<
    SessionRecord,
    'sessionId' | 'anonymousUserId' | 'consentVersion' | 'consentedAt'
  >,
  now = Date.now(),
): Promise<ResearchSessionVerification> {
  await initializeDatabase(db);
  const token = cookieValue(request, RESEARCH_SESSION_COOKIE);
  if (!token || !/^[A-Za-z0-9_-]{43}$/u.test(token)) {
    return { ok: false, error: 'A valid anonymous research session is required' };
  }
  const tokenHash = await sha256Hex(token);
  const row = await db
    .prepare(
      `SELECT token_hash, session_id, anonymous_user_id, consent_version,
              consented_at, issued_at, expires_at, revoked_at
       FROM research_session_tokens WHERE token_hash = ?`,
    )
    .bind(tokenHash)
    .first<ResearchSessionTokenRow>();
  return validateStoredResearchSession(row, expected, now);
}
