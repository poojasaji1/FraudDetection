CREATE TABLE IF NOT EXISTS predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id VARCHAR,
  amount FLOAT,
  fraud_score FLOAT,
  label VARCHAR,
  latency_ms FLOAT,
  model_version VARCHAR,
  all_features JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  checked_at TIMESTAMPTZ DEFAULT NOW(),
  drift_detected BOOLEAN,
  feature_psi JSONB,
  score_kl FLOAT,
  triggered_features JSONB
);
