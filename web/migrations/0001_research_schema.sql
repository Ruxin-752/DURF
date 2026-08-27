CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  anonymous_user_id TEXT NOT NULL,
  consent_version TEXT NOT NULL,
  consented_at INTEGER NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  client_version TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  summary_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  sequence_number INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  occurred_at INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  model_hash TEXT,
  route_trace TEXT,
  probabilities_json TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_session_sequence
  ON events(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  session_id TEXT NOT NULL,
  utterance TEXT NOT NULL,
  route TEXT NOT NULL,
  top_label TEXT NOT NULL,
  low_confidence INTEGER NOT NULL,
  probabilities_json TEXT NOT NULL,
  phrases_json TEXT NOT NULL,
  model_hash TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  route_trace TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(event_id),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_top_label ON feedback(top_label);
