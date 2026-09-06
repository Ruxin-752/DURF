'use client';

import { useEffect, useState, type FormEvent } from 'react';
import { loadSyntheticGameModels, type GameModels, type GroundingPrediction } from '@/lib/synthetic-browser-models';
import { splitFeedbackPhrases } from '@/lib/route-inputs';
import type { FeedbackFormPrediction } from '@/lib/browser-models';

interface Result { text: string; speech: FeedbackFormPrediction; grounding: GroundingPrediction }

export default function ModelCheckPage() {
  const [models, setModels] = useState<GameModels | null>(null);
  const [error, setError] = useState('');
  const [input, setInput] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  useEffect(() => {
    let active = true;
    loadSyntheticGameModels().then((loaded) => { if (active) setModels(loaded); })
      .catch(() => { if (active) setError('The models could not load. Please reload.'); });
    return () => { active = false; };
  }, []);
  function analyze(event: FormEvent) {
    event.preventDefault();
    if (!models || !input.trim()) return;
    setResults(splitFeedbackPhrases(input).map((text) => ({
      text, speech: models.classify(text), grounding: models.ground(text),
    })));
  }
  return (
    <main style={{ width: 'calc(100% - 32px)', maxWidth: 860, margin: '32px auto', padding: 24,
      background: '#fff7e8', color: '#35251f', border: '2px solid #a8895e', borderRadius: 12,
      fontFamily: 'system-ui, sans-serif', lineHeight: 1.6 }}>
      <p><a href="/" style={{ color: '#754322', textDecoration: 'underline' }}>Back to the kitchen</a></p>
      <h1 style={{ fontSize: 26, fontWeight: 700, margin: '16px 0' }}>Feedback classification check</h1>
      <p>English · trained and tested on synthetic examples. Human performance has not been measured.</p>
      <p>Your text stays in this browser. This page does not submit research data.</p>
      <form onSubmit={analyze} className="feedback-form" style={{ margin: '20px 0' }}>
        <label htmlFor="check-text">Feedback to check</label>
        <textarea id="check-text" maxLength={500} rows={5} value={input}
          onChange={(event) => setInput(event.target.value)} placeholder="Enter an English sentence or several clauses." />
        <button disabled={!models || !input.trim()}>{models ? 'Analyze' : 'Loading models…'}</button>
      </form>
      {error && <p role="alert">{error}</p>}
      <section aria-live="polite" aria-label="Classification results">
        {results.map(({ text, speech, grounding }, index) => (
          <article key={`${index}:${text}`} className="phrase-card" style={{ marginTop: 16, padding: 16 }}>
            <blockquote>{text}</blockquote>
            <h2 style={{ fontSize: 20, fontWeight: 700 }}>{speech.label}{speech.abstained ? ' · uncertain' : ''}</h2>
            <p>Raw model scores, not probability of correctness:</p>
            <ul>{Object.entries(speech.probabilities).map(([label, value]) => (
              <li key={label}>{label}: {(value * 100).toFixed(1)}%</li>
            ))}</ul>
            <p>Independent reward reference: <strong>{grounding.label}</strong>
              {' '}({(grounding.confidence * 100).toFixed(1)}% raw score{grounding.abstained ? ', uncertain' : ''}).
              This reference determines where Route1 looks for features; it is separate from the visible speech-act label.</p>
          </article>
        ))}
      </section>
      {models && <small>Release: {models.classifierEvidence.modelVersion}</small>}
    </main>
  );
}
