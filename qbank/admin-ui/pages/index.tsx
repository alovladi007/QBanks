import { useState } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';
type ItemRow = { question_id:number; version:number; topic_id?:number; sh_p:number; recent_attempts:number };
export default function Admin() {
  const [token, setToken] = useState(''); const [items, setItems] = useState<ItemRow[]>([]);
  const headers = { 'Content-Type':'application/json', 'Authorization': `Bearer ${token}` };
  const fetchItems = async () => { const r = await fetch(`${API}/v1/admin/exposure/items?limit=200`, { headers }); setItems(await r.json()); };
  const setSh = async (qid:number, ver:number, sh_p:number) => { await fetch(`${API}/v1/admin/exposure/set`, { method:'POST', headers, body: JSON.stringify({ question_id: qid, version: ver, sh_p }) }); await fetchItems(); };
  return (<main style={{padding:24}}>
    <h1>Admin: Sympson–Hetter</h1>
    <p>Paste an <b>admin</b> JWT</p>
    <textarea value={token} onChange={(e)=>setToken(e.target.value)} rows={4} style={{width:'100%'}} />
    <div style={{marginTop:12}}><button onClick={fetchItems}>Load</button></div>
    <table style={{marginTop:16, width:'100%', borderCollapse:'collapse'}}>
      <thead><tr><th>QID</th><th>Ver</th><th>Topic</th><th>sh_p</th><th>Attempts(7d)</th><th>Save</th></tr></thead>
      <tbody>{items.map(it => (
        <tr key={`${it.question_id}-${it.version}`} style={{borderTop:'1px solid #eee'}}>
          <td>{it.question_id}</td><td>{it.version}</td><td>{it.topic_id ?? ''}</td>
          <td><input type="number" min="0" max="1" step="0.05" defaultValue={it.sh_p} onBlur={(e)=>setSh(it.question_id, it.version, parseFloat(e.target.value))} /></td>
          <td>{it.recent_attempts}</td><td><button onClick={()=>setSh(it.question_id, it.version, it.sh_p)}>Save</button></td>
        </tr>))}</tbody>
    </table></main>);
}