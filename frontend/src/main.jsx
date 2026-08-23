import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function App() {
  const [session, setSession] = useState(null);
  const [message, setMessage] = useState('Find me a black backpack suitable for university under $80.');
  const [data, setData] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api('/model-info').then(setStatus).catch(error => setError(error.message));
  }, []);

  async function getSession() {
    if (session) return session;
    const created = await api('/sessions', {method: 'POST'});
    setSession(created.id);
    return created.id;
  }

  async function search(event) {
    event.preventDefault();
    setBusy(true); setError('');
    try {
      const id = await getSession();
      setData(await api(`/sessions/${id}/search`, {
        method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({message}),
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function feedback(productId, kind) {
    try {
      await api(`/sessions/${session}/feedback`, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({product_id: productId, kind})});
    } catch (requestError) { setError(requestError.message); }
  }

  return <main>
    <header>
      <div><p className="eyebrow">SFT + RAG + agent memory</p><h1>Adaptive Product Search</h1></div>
      <span className={status ? 'status online' : 'status'}>{status ? `LoRA online · ${status.model_version}` : 'LoRA unavailable'}</span>
    </header>
    <p className="lede">BM25 retrieves candidates. A Qwen LoRA adapter trained on Amazon ESCI makes the final Exact / Substitute / Complement / Irrelevant judgments.</p>
    <form onSubmit={search}>
      <input aria-label="Shopping request" value={message} onChange={event => setMessage(event.target.value)} />
      <button disabled={busy}>{busy ? 'Asking adapter…' : 'Search'}</button>
    </form>
    {error && <div className="error"><strong>Fine-tuned model required.</strong> {error}</div>}
    {data && <>
      <section className="answer"><p className="eyebrow">Grounded answer</p><p>{data.answer}</p><small>{data.citations.map(citation => `[${citation.rank}] ${citation.product_id}`).join(' · ') || 'No citations returned'}</small></section>
      <div className="meta"><span>Model: {data.model_version}</span><span>{data.latency_ms} ms</span><span>{data.results.length} results</span></div>
      {data.results.length === 0 && <div className="empty">No candidates met the current hard constraints. Try broadening the request.</div>}
      <div className="grid">{data.results.map((row, index) => <article key={row.product.id}>
        <div className="rank">{index + 1}</div>
        <span className={`label label-${row.relevance_label}`}>{row.relevance_label}</span>
        <h2>{row.product.title}</h2>
        <p className="product-meta">{row.product.brand || 'Brand unavailable'} · {row.product.price == null ? 'Price unavailable in ESCI' : `$${row.product.price}`}</p>
        <p>{row.explanation}</p>
        <details><summary>Scores</summary><pre>{JSON.stringify(row.scores, null, 2)}</pre></details>
        <div className="actions"><button onClick={() => feedback(row.product.id, 'like')}>Like</button><button onClick={() => feedback(row.product.id, 'not_relevant')}>Not relevant</button></div>
      </article>)}</div>
      <details className="trace"><summary>Agent trace and memory</summary><pre>{JSON.stringify({intent: data.parsed_intent, remembered: data.persistent_constraints, tools: data.tools_selected, trace_id: data.trace_id}, null, 2)}</pre></details>
    </>}
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
