CREATE TABLE IF NOT EXISTS research_session_tokens (
  token_hash TEXT PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  anonymous_user_id TEXT NOT NULL,
  consent_version TEXT NOT NULL,
  consented_at INTEGER NOT NULL,
  issued_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_research_session_tokens_expiry
  ON research_session_tokens(expires_at);

CREATE TABLE IF NOT EXISTS research_rate_limits (
  bucket_key TEXT NOT NULL,
  window_start INTEGER NOT NULL,
  request_count INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (bucket_key, window_start)
);

CREATE INDEX IF NOT EXISTS idx_research_rate_limits_expiry
  ON research_rate_limits(expires_at);

CREATE TABLE IF NOT EXISTS research_security_secrets (
  secret_name TEXT PRIMARY KEY,
  secret_value TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
