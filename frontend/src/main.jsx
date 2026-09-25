import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === 'object' ? payload.detail?.message : payload.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

function score(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function App() {
  const [query, setQuery] = useState('makita impact drill');
  const [data, setData] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api('/ready').then(setStatus).catch(requestError => setError(requestError.message));
  }, []);

  async function search(event) {
    event.preventDefault();
    setBusy(true); setError('');
    try {
      setData(await api('/search', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({query}),
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return <main>
    <header>
      <div><p className="eyebrow">Hybrid retrieval · learning to rank</p><h1>Adaptive Product Search</h1></div>
      <span className={status ? 'status online' : 'status'}>{status ? 'Search ready' : 'Search unavailable'}</span>
    </header>
    <p className="lede">Query understanding normalizes the request for indexed BM25 and MiniLM retrieval; a MiniLM cross-encoder makes the final ranking decision.</p>
    <form onSubmit={search}>
      <input aria-label="Product search query" value={query} onChange={event => setQuery(event.target.value)} />
      <button disabled={busy}>{busy ? 'Ranking…' : 'Search'}</button>
    </form>
    {error && <div className="error"><strong>Search unavailable.</strong> {error}</div>}
    {data && <>
      <div className="meta"><span>{data.latency_ms} ms</span><span>{data.candidate_count} candidates reranked</span><span>{data.results.length} results</span></div>
      {data.results.length === 0 && <div className="empty">No catalog matches were found.</div>}
      <div className="grid">{data.results.map((row, index) => <article key={row.product.id}>
        <div className="rank">{index + 1}</div>
        <span className={`label label-${row.relevance_label}`}>{row.relevance_label}</span>
        <h2>{row.product.title}</h2>
        <p className="product-meta">{row.product.brand || 'Brand unavailable'} · {row.product.price == null ? 'Price unavailable in ESCI' : `$${row.product.price}`}</p>
        <p>{row.explanation}</p>
        <small>Expected gain: {score(row.scores.cross_encoder_expected_gain, 2)} / 3</small>
      </article>)}</div>
      <details className="debug-panel">
        <summary>Search-quality details</summary>
        <div className="debug-grid">
          <section>
            <p className="eyebrow">Query analysis</p>
            <dl>
              <dt>Normalized</dt><dd>{data.query_analysis.normalized_query}</dd>
              <dt>Brands</dt><dd>{data.query_analysis.brands.join(', ') || '—'}</dd>
              <dt>Colours</dt><dd>{data.query_analysis.colors.join(', ') || '—'}</dd>
              <dt>Model tokens</dt><dd>{data.query_analysis.model_tokens.join(', ') || '—'}</dd>
              <dt>Exclusions</dt><dd>{data.query_analysis.exclusions.join(', ') || '—'}</dd>
            </dl>
          </section>
          <section>
            <p className="eyebrow">Adaptive retrieval</p>
            <dl>
              <dt>BM25 weight</dt><dd>{score(data.retrieval_decision.bm25_weight, 2)}</dd>
              <dt>Dense weight</dt><dd>{score(data.retrieval_decision.dense_weight, 2)}</dd>
              <dt>Lexical rarity</dt><dd>{score(data.retrieval_decision.lexical_rarity, 2)}</dd>
              <dt>Decision</dt><dd>{data.retrieval_decision.reasons.join(' · ')}</dd>
            </dl>
          </section>
        </div>
        <div className="debug-results" role="region" aria-label="Ranking signal details" tabIndex="0">
          <table>
            <thead><tr><th>Rank</th><th>Cross-encoder</th><th>BM25</th><th>Dense</th><th>Fusion</th><th>Field boost</th><th>Exclusion</th></tr></thead>
            <tbody>{data.results.map((row, index) => <tr key={row.product.id}>
              <td>{index + 1}</td>
              <td>{score(row.scores.cross_encoder_expected_gain, 2)} ({row.relevance_label})</td>
              <td>#{row.scores.bm25_rank || '—'}</td>
              <td>#{row.scores.dense_rank || '—'}</td>
              <td>{score(row.scores.fusion_score, 4)}</td>
              <td>{score(row.scores.field_lexical_boost, 2)}</td>
              <td>{score(row.scores.negation_penalty, 2)}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </details>
    </>}
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
