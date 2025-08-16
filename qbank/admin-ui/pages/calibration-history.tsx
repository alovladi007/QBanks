import { useEffect, useMemo, useState } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';
type RunRow = { id:string; exam_code:string; status:string; created_at:string; started_at?:string; finished_at?:string };
type DiffRow = { qid:number; ver:number; before:number; after:number; delta:number };
type RunDetail = { id:string; exam_code:string; status:string; params:any; history:any[]; result?:{updated:number; history:any[]; diff:DiffRow[]}; error?:string; created_at:string; started_at?:string; finished_at?:string };
function toCSV(rows: DiffRow[]) { const h="qid,ver,sh_p_before,sh_p_after,delta\n"; const b=rows.map(r=>[r.qid,r.ver,r.before.toFixed(6),r.after.toFixed(6),r.delta.toFixed(6)].join(",")).join("\n"); return h+b+"\n"; }
function downloadCSV(filename:string, content:string){ const blob=new Blob([content],{type:"text/csv;charset=utf-8;"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url); }
function LineChart({ points, width=520, height=180 }:{ points: {x:number;y:number}[]; width?:number;height?:number }) {
  if (!points.length) return <svg width={width} height={height} />;
  const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
  const minX=Math.min(...xs), maxX=Math.max(...xs); const minY=Math.min(...ys,0), maxY=Math.max(...ys,0.001); const pad=24;
  const sx=(x:number)=> pad + ((x-minX)/Math.max(1,(maxX-minX)))*(width-2*pad);
  const sy=(y:number)=> height - pad - ((y-minY)/Math.max(1e-9,(maxY-minY)))*(height-2*pad);
  const path=points.map((p,i)=> (i===0?`M ${sx(p.x)} ${sy(p.y)}`:`L ${sx(p.x)} ${sy(p.y)}`)).join(" ");
  const xTicks=Array.from(new Set(points.map(p=>p.x))); const yTicks=[minY,(minY+maxY)/2,maxY];
  return (<svg width={width} height={height}>
    <rect x={0} y={0} width={width} height={height} fill="#fff" stroke="#e5e7eb" />
    <line x1={pad} y1={height-pad} x2={width-pad} y2={height-pad} stroke="#9ca3af" />
    <line x1={pad} y1={pad} x2={pad} y2={height-pad} stroke="#9ca3af" />
    {xTicks.map((t,i)=>(<text key={i} x={sx(t)} y={height-pad+12} fontSize={10} textAnchor="middle">{t}</text>))}
    {yTicks.map((t,i)=>(<g key={i}><line x1={pad-4} y1={sy(t)} x2={pad} y2={sy(t)} stroke="#9ca3af" /><text x={4} y={sy(t)} fontSize={10} dominantBaseline="middle">{t.toFixed(3)}</text></g>))}
    <path d={path} fill="none" stroke="#4a90e2" strokeWidth={2} />
    {points.map((p,i)=>(<circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.5} fill="#1f77b4" />))}
    <text x={pad} y={16} fontSize={12} fontWeight={600}>max_over vs iteration</text>
  </svg>);
}
export default function CalibHistory() {
  const [token, setToken] = useState(''); const [runs, setRuns] = useState<RunRow[]>([]); const [selected, setSelected] = useState<RunDetail | null>(null);
  const [filters, setFilters] = useState({ exam_code:'', start:'', end:'', page:1, page_size:25 });
  const headers = { 'Content-Type':'application/json', 'Authorization': `Bearer ${token}` };
  const loadRuns = async () => { const qs = new URLSearchParams(); if (filters.exam_code) qs.set('exam_code', filters.exam_code); if (filters.start) qs.set('start', filters.start); if (filters.end) qs.set('end', filters.end); qs.set('page', String(filters.page)); qs.set('page_size', String(filters.page_size)); const r = await fetch(`${API}/v1/admin/exposure/calibrate_sh/runs?`+qs.toString(), { headers }); if (r.ok) setRuns(await r.json()); };
  const loadRun = async (id:string) => { const r = await fetch(`${API}/v1/admin/exposure/calibrate_sh/runs/${id}`, { headers }); if (r.ok) setSelected(await r.json()); };
  useEffect(()=>{ if (token) loadRuns(); }, [token]);
  useEffect(()=>{ if (token) loadRuns(); }, [filters.exam_code, filters.start, filters.end, filters.page, filters.page_size]);
  const points = useMemo(()=> selected?.history?.map((h:any)=> ({ x: h.iter, y: Number(h.max_over || 0) })) || [], [selected?.history]);
  const exportCSV = () => { if (!selected?.result?.diff?.length) return; const csv = toCSV(selected.result.diff); downloadCSV(`calibration_diff_${selected.id}.csv`, csv); };
  return (<main style={{padding:24, display:'grid', gridTemplateColumns:'1fr 1fr', gap:24}}>
    <section>
      <h1>Calibration Runs</h1><p>Paste an <b>admin</b> JWT</p>
      <textarea value={token} onChange={(e)=>setToken(e.target.value)} rows={3} style={{width:'100%'}} />
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:12}}>
        <label>Exam <input value={filters.exam_code} onChange={(e)=>setFilters({...filters, exam_code:e.target.value, page:1})} /></label>
        <label>Page size <input type="number" value={filters.page_size} onChange={(e)=>setFilters({...filters, page_size:parseInt(e.target.value)||25, page:1})}/></label>
        <label>Start (ISO) <input placeholder="2025-08-01" value={filters.start} onChange={(e)=>setFilters({...filters, start:e.target.value, page:1})} /></label>
        <label>End (ISO) <input placeholder="2025-08-31" value={filters.end} onChange={(e)=>setFilters({...filters, end:e.target.value, page:1})} /></label>
      </div>
      <div style={{marginTop:8}}>
        <button onClick={()=>setFilters({...filters, page: Math.max(1, filters.page-1)})}>Prev</button>
        <span style={{margin:'0 8px'}}>Page {filters.page}</span>
        <button onClick={()=>setFilters({...filters, page: filters.page+1})}>Next</button>
        <button style={{marginLeft:12}} onClick={loadRuns}>Refresh</button>
      </div>
      <table style={{marginTop:12, width:'100%', borderCollapse:'collapse'}}>
        <thead><tr><th>Started</th><th>Exam</th><th>Status</th><th>Run</th></tr></thead>
        <tbody>{runs.map(r => (<tr key={r.id} style={{borderTop:'1px solid #eee'}}>
          <td>{r.started_at || r.created_at}</td><td>{r.exam_code}</td><td>{r.status}</td><td><button onClick={()=>loadRun(r.id)}>View</button></td>
        </tr>))}</tbody>
      </table>
    </section>
    <section>
      <h1>Details</h1>
      {!selected && <p>Select a run</p>}
      {selected && (<div>
        <p><b>ID:</b> {selected.id}</p><p><b>Exam:</b> {selected.exam_code} — <b>Status:</b> {selected.status}</p>
        <p><b>Window:</b> {selected.started_at} → {selected.finished_at}</p>
        <h3>max_over chart</h3><LineChart points={points} />
        <h3 style={{marginTop:12}}>Params</h3><pre style={{background:'#fafafa', padding:12, maxHeight:180, overflow:'auto'}}>{JSON.stringify(selected.params, null, 2)}</pre>
        <h3>History</h3><pre style={{background:'#fafafa', padding:12, maxHeight:200, overflow:'auto'}}>{JSON.stringify(selected.history, null, 2)}</pre>
        <h3>Diff (before/after sh_p)</h3>
        <div style={{display:'flex', gap:8, alignItems:'center'}}><button onClick={exportCSV}>Export CSV</button><span style={{color:'#666'}}>rows: {selected.result?.diff?.length || 0}</span></div>
        <pre style={{background:'#fafafa', padding:12, maxHeight:200, overflow:'auto'}}>
{`qid,ver,sh_p_before,sh_p_after,delta
`}{selected.result?.diff?.slice(0,10)?.map((d:any)=>`${d.qid},${d.ver},${d.before.toFixed(4)},${d.after.toFixed(4)},${d.delta.toFixed(4)}`).join("\n")}
{selected.result?.diff?.length>10 ? "\n… (see CSV for full list)" : ""}
        </pre>
        {selected.error && (<><h3>Error</h3><pre style={{background:'#fff0f0', padding:12}}>{selected.error}</pre></>)}
      </div>)}
    </section>
  </main>);
}