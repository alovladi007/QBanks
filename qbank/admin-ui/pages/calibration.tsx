import { useEffect, useState } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';
type Status = { state:string; current_iter:number; total_iters:number; avg_exp?:number; max_over?:number; result?:any };
export default function Calibration() {
  const [token, setToken] = useState(''); const [jobId, setJobId] = useState<string>(''); const [status, setStatus] = useState<Status | null>(null);
  const [form, setForm] = useState({ exam_code:'DEMO-EXAM', tau:0.2, n:400, test_len:25, iters:5, alpha:0.6 });
  const headers = { 'Content-Type':'application/json', 'Authorization': `Bearer ${token}` };
  const start = async () => { const r = await fetch(`${API}/v1/admin/exposure/calibrate_sh/start`, { method:'POST', headers, body: JSON.stringify(form) }); const data = await r.json(); setJobId(data.job_id); };
  useEffect(()=>{ const t = setInterval(async () => { if (!jobId) return; const r = await fetch(`${API}/v1/admin/exposure/calibrate_sh/status?job_id=${jobId}`, { headers }); if (r.ok) setStatus(await r.json()); }, 1500); return () => clearInterval(t); }, [jobId, token]);
  const pct = status?.total_iters ? Math.round(100 * (status!.current_iter / status!.total_iters)) : 0;
  return (<main style={{padding:24, maxWidth:800}}>
    <h1>Calibration</h1><p>Paste an <b>admin</b> JWT</p>
    <textarea value={token} onChange={(e)=>setToken(e.target.value)} rows={3} style={{width:'100%'}} />
    <section style={{marginTop:16}}><h3>Parameters</h3>
      <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:12}}>
        <label>Exam <input value={form.exam_code} onChange={(e)=>setForm({...form, exam_code:e.target.value})} /></label>
        <label>τ <input type="number" step="0.01" value={form.tau} onChange={(e)=>setForm({...form, tau:parseFloat(e.target.value)})} /></label>
        <label>n <input type="number" value={form.n} onChange={(e)=>setForm({...form, n:parseInt(e.target.value)})} /></label>
        <label>length <input type="number" value={form.test_len} onChange={(e)=>setForm({...form, test_len:parseInt(e.target.value)})} /></label>
        <label>iters <input type="number" value={form.iters} onChange={(e)=>setForm({...form, iters:parseInt(e.target.value)})} /></label>
        <label>α <input type="number" step="0.1" value={form.alpha} onChange={(e)=>setForm({...form, alpha:parseFloat(e.target.value)})} /></label>
      </div><button style={{marginTop:12}} onClick={start}>Start</button>
    </section>
    {status && (<section style={{marginTop:24}}>
      <h3>Status: {status.state}</h3>
      <div style={{height:16, background:'#eee', borderRadius:8, overflow:'hidden'}}><div style={{width:`${pct}%`, height:'100%', background:'#4a90e2'}} /></div>
      <p style={{marginTop:8}}>iter {status.current_iter} / {status.total_iters} · avg_exp {status.avg_exp?.toFixed(3)} · max_over {status.max_over?.toFixed(3)}</p>
      {status.result && <pre style={{maxHeight:200, overflow:'auto', background:'#fafafa', padding:12}}>{JSON.stringify(status.result, null, 2)}</pre>}
    </section>)}
  </main>);
}