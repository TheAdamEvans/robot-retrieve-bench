import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {fileURLToPath} from 'node:url';
import {marked} from 'marked';
import {chromium} from 'playwright';

const root=path.dirname(fileURLToPath(import.meta.url));
const assetRoot=path.join(root,'assets');
const data=JSON.parse(await fs.readFile(path.join(assetRoot,'data.json'),'utf8'));
const source=await fs.readFile(path.join(root,'story.md'),'utf8');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const mime={'.jpg':'image/jpeg','.mp4':'video/mp4','.svg':'image/svg+xml','.json':'application/json','.md':'text/markdown','.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2'};
const uri=async p=>`data:${mime[path.extname(p)]||'application/octet-stream'};base64,${(await fs.readFile(p)).toString('base64')}`;
const asset=async name=>uri(path.join(assetRoot,name));
const img=async(name,alt,cls='')=>`<img class="${cls}" src="${await asset(name)}" alt="${esc(alt)}" loading="lazy" decoding="async">`;
const video=async(name,alt)=>`<video controls muted playsinline preload="none" poster="${await asset(name+'.jpg')}" aria-label="${esc(alt)}"><source src="${await asset(name+'.mp4')}" type="video/mp4">${esc(alt)}</video>`;
const fixed=x=>Number(x).toFixed(2);
const opt=m=>m&&m.mean!=null?fixed(m.mean):'–';
const shortStatus=s=>s.replace('ANSWERED_','').replace('INSUFFICIENT_EVIDENCE','INSUFFICIENT').replace('NONE_FOUND_EXHAUSTIVE','NONE FOUND').toLowerCase();

const components={
  hero:async()=>`<figure class="hero-image">${await img('hero.jpg','Front camera during the Library-to-MLK recording at 30 seconds')}<span class="frame-corner">RECORDED EVIDENCE / SCAND</span><figcaption><span><strong>A doorway is a scene.<br>A traversal is an event.</strong></span><span>LIBRARY → MLK<br>FRONT CAMERA · t = 30.00 s</span></figcaption></figure>`,
  stats:async()=>`<div class="stats"><div class="stat"><b>${data.corpus.recordings}</b><span>recordings · two robots · 2 held out</span></div><div class="stat"><b>${Math.round(data.corpus.seconds).toLocaleString()} s</b><span>${(data.corpus.raw_bytes/1e9).toFixed(0)} GB of raw logs</span></div><div class="stat"><b>${data.corpus.judgments.toLocaleString()}</b><span>relevance judgments</span></div><div class="stat"><b>32 × 10</b><span>questions × configurations</span></div></div>`,
  corpus:async()=>`<div class="corpus-grid"><figure>${await img('butler.jpg','Spot front-camera frame from Butler at 50 seconds')}<figcaption>SPOT · BUTLER → LBJ · t = 50 s</figcaption></figure><figure>${await img('sanjac.jpg','Jackal front-camera frame from Sanjac at 70 seconds')}<figcaption>JACKAL · SANJAC → STADIUM · t = 70 s</figcaption></figure></div>`,
  measures:async()=>`<div class="measures">${[['01 / RELEVANCE','Does the ranking help?','nDCG@10 and pooled recall; judged coverage stays visible.'],['02 / CONSTRAINTS','Did the clauses hold?','Program validity, status agreement, and explicit unknowns.'],['03 / COST','What did the answer cost?','Latency, model tokens, and instrumented reads at each stage.'],['04 / CONFIDENCE','How much should we trust it?','Intent-level uncertainty and an independent-session label re-check.']].map(([label,title,body])=>`<div class="measure"><span class="label">${label}</span><b>${title}</b><p>${body}</p></div>`).join('')}</div>`,
  explorer:async()=>{
    const plots=[];
    for(const split of ['compose_test','demo5_para','demo5','l1_compositions_dev'])for(const metric of ['ndcg10','roc_pen'])for(const mode of ['fresh','cached']){
      const key=`${split}-${metric}-${mode}`;
      plots.push(`<img class="plot" data-plot="${key}" ${key==='compose_test-ndcg10-fresh'?'':'hidden'} src="${await asset('quality-'+key+'.svg')}" alt="${esc(split+' '+metric+' quality estimates with confidence intervals and '+mode+' median latency')}">`);
    }
    return `<div class="explorer"><div class="explorer-head"><div><h3>Explore the operating tradeoff</h3><p>Frozen eval v9 checkpoint · select a slice, measure, and execution mode</p></div><span class="live-indicator">INTERACTIVE / OFFLINE</span></div>
    <div class="controls"><label for="slice">Question set<select id="slice"><option value="compose_test">Compositions + controls</option><option value="demo5_para">Paraphrases</option><option value="demo5">Original demo questions</option><option value="l1_compositions_dev">Behavior questions (complete ground truth)</option></select></label><label for="metric">Quality measure<select id="metric"><option value="ndcg10">nDCG@10 · ranking quality</option><option value="roc_pen">ROC-AUC · judged-set ordering</option></select></label><label for="latency">Program generation<select id="latency"><option value="fresh">Fresh program · estimated total</option><option value="cached">Cached program · measured replay</option></select></label></div>
    ${plots.join('')}<p class="chart-note" id="chart-note"></p><div class="table-wrap"><table><thead><tr><th>Approach</th><th id="metric-title">nDCG@10</th><th>Median</th><th>Mean LLM tokens</th><th>Judged@50</th></tr></thead><tbody id="metric-table"></tbody></table></div>
    <div class="budget-box"><label for="budget">Median-latency budget<select id="budget"><option value="100">100 ms</option><option value="1000">1 second</option><option value="10000" selected>10 seconds</option></select></label><p id="budget-result" aria-live="polite"></p></div></div>`;
  },
  doorway:async()=>`<div class="evidence-grid"><div class="evidence-card"><div class="card-top"><b><span class="swatch" style="background:#3475a5"></span>EMBED · first result</b><span class="badge grade0">GRADE 0 / NOT RELEVANT</span></div>${await video('doorway-embed','EMBED first result, Library MLK from 114 to 118 seconds')}<div class="card-body"><code>Library_MLK:0118</code> · 114–118 s<br>The retrieved window is graded non-relevant to this doorway intent. Search status: unverified.</div></div><div class="evidence-card"><div class="card-top"><b><span class="swatch" style="background:#087f72"></span>FUSED · first result</b><span class="badge">GRADE 2 / RELEVANT</span></div>${await video('doorway-fused','FUSED first result, Library MLK from 25 to 29 seconds')}<div class="card-body"><code>Library_MLK:0029</code> · 25–29 s<br>The label pool grades this window relevant. The retrieval itself supplies no clause checks. Search status: unverified.</div></div></div>`,
  'program-diff':async()=>{
    const examples=data.examples.filter(x=>x.query_id==='test_turn_then_brake:canonical'&&['PROGRAM_LUNA','PROGRAM_ORACLE'].includes(x.config));
    return `<div class="table-wrap"><table><thead><tr><th>Interpretation</th><th>Temporal relation</th><th>First mapped window</th><th>L2 grade</th></tr></thead><tbody>${examples.map(x=>`<tr><td>${x.config==='PROGRAM_LUNA'?'Generated':'Reference'}</td><td>turn.${x.program.relations[0].parent.point} → brake.START<br>AFTER, within 5 s</td><td>${x.windows[0].id}</td><td>${x.windows[0].grade}</td></tr>`).join('')}</tbody></table></div>`;
  },
  receipt:async()=>{
    const labels={'/image_raw/compressed':'Front camera','/spot/camera/frontleft/image/compressed':'Body · front-left','/spot/camera/frontright/image/compressed':'Body · front-right','/spot/camera/left/image/compressed':'Body · left','/spot/camera/right/image/compressed':'Body · right','/spot/camera/back/image/compressed':'Body · rear','/velodyne_points':'Lidar','/odom':'Odometry','/tf':'Transform'};
    return `<div class="receipt-layout"><div class="receipt-media">${await video('receipt','Recorded acceleration onset, with an anchor marker at the evidence cutoff')}<p class="clip-caption">LIBRARY_MLK · CUTOFF ${data.receipt.event_s.toFixed(3)} s</p>${await img('motion.svg','Measured odometry speed from 22 to 33 seconds with the receipt cutoff marked','motion')}<p class="note">Measured smoothed speed. The red line is the strict-before cutoff; the clip flashes an anchor marker.</p></div><div><table class="receipt-table"><thead><tr><th>Sensor</th><th>Age</th><th>Receipt check</th></tr></thead><tbody>${data.receipt.items.map(x=>`<tr><td>${labels[x.topic]||esc(x.topic)}</td><td>${(Number(x.ageNs)/1e6).toFixed(1)} ms</td><td class="${x.ok==='TRUTH_TRUE'?'ok':'unknown'}">${x.ok==='TRUTH_TRUE'?'Valid':'Unknown · future stamp'}</td></tr>`).join('')}</tbody></table><p class="note">Ages use measurement timestamps relative to the cutoff. Selection also checks when the message was available.</p></div></div>`;
  },
  'receipt-json':async()=>`<pre><code>${esc(JSON.stringify(data.receipt,null,2))}</code></pre>`,
  generalization:async()=>{
    const g=data.generalization, cfgs=['EMBED','FUSED','FUSED_V_LUNA','PROGRAM_LUNA','PROGRAM_ORACLE'];
    return `${await img('generalization.svg','Composition nDCG@10 on the original seven recordings, the nine new recordings, and the two held-out SCAND Val recordings','motion')}<div class="table-wrap"><table><thead><tr><th>Approach</th>${Object.keys(g).map(k=>`<th>${esc(k)}<br><small>nDCG@10 / AUC</small></th>`).join('')}</tr></thead><tbody>${cfgs.map(c=>`<tr><td><span class="swatch" style="background:${data.colors[c]}"></span>${data.names[c]}${c==='PROGRAM_ORACLE'?' †':''}</td>${Object.values(g).map(v=>`<td>${fixed(v[c].ndcg10)} / ${fixed(v[c].roc_pen)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="note">Composition questions, split by where each judged window comes from. † Oracle uses a supplied program.</p>`;
  },
  dream:async()=>{
    const d=data.dream, cfgs=['EMBED','FUSED','PROGRAM_LUNA','FUSED_V_LUNA'];
    const truth=q=>q.full+q.partial===0?`<b>none in scope</b><br><small>${q.proof==='numeric screen'?'no span passes the numeric clauses':'every candidate judged'}</small>`:`<b>${q.full}</b> full · ${q.partial} partial`;
    const cell=(q,c)=>{const f=q.found[c];if(!f)return '<td>–</td>';return q.full+q.partial===0?`<td class="unknown">${shortStatus(f.status)}</td>`:`<td>${q.full?`${f.full10}/${q.full} · ${f.full50}/${q.full}`:'–'}<br><small>${f.rel50}/${q.full+q.partial} relevant in top 50</small></td>`;};
    return `<div class="table-wrap"><table><thead><tr><th>Question</th><th>Complete ground truth</th>${cfgs.map(c=>`<th><span class="swatch" style="background:${data.colors[c]}"></span>${data.names[c]}</th>`).join('')}</tr></thead><tbody>${d.questions.map(q=>`<tr><td>${esc(q.title)}</td><td>${truth(q)}</td>${cfgs.map(c=>cell(q,c)).join('')}</tr>`).join('')}</tbody></table></div><p class="note">Cells: full matches found in the top 10 · top 50, then relevant (full or partial) episodes in the top 50. For the two questions with nothing in scope, the cell shows the status each system returned: a correct system would say none found.</p>`;
  },
  uncertainty:async()=>`<div class="status-grid"><div class="status-item"><h3>“When did the battery drop below 20%?”</h3><span class="badge grade1">PROGRAM / INSUFFICIENT EVIDENCE</span><span class="badge">FUSED / UNVERIFIED</span><p>Missing support for the requested signal. A ranked visual result does not resolve the question.</p></div><div class="status-item"><h3>“A person in a side camera, nobody in front.”</h3><span class="badge grade1">FUSED + VERIFY / PARTIAL</span><span class="badge grade1">PROGRAM / INSUFFICIENT EVIDENCE</span><p>The required body-camera person clause is UNKNOWN / NOT_INDEXED. The missing clause stays visible.</p></div></div>`,
  downloads:async()=>`<div class="downloads"><a class="download" download="fleet-search-story.md" href="${await uri(path.join(root,'story.md'))}">Download Markdown source</a><a class="download" download="benchmark-v9.json" href="${await uri(path.join(root,'../benchmark/results/v9/report.json'))}">Download benchmark JSON</a><a class="download" download="benchmark-v9.html" href="${await uri(path.join(root,'../benchmark/results/v9/report.html'))}">Download full HTML report</a></div>`,
  'full-results':async()=>Object.entries(data.report.tables).map(([split,rows])=>`<p><strong>${esc(split)}</strong></p><div class="table-wrap"><table class="full-results"><thead><tr><th>Configuration</th><th>nDCG@10</th><th>AUC</th><th>R@50</th><th>Judged@50</th><th>Median ms*</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.config)}</td><td>${opt(r.ndcg10)}</td><td>${opt(r.roc_pen)}</td><td>${opt(r.r50)}</td><td>${r.judged50?(100*r.judged50.mean).toFixed(0)+'%':'–'}</td><td>${r.wall_p50==null?'–':r.wall_p50.toFixed(0)}</td></tr>`).join('')}</tbody></table></div>`).join('')+'<p class="note">* Fresh-program estimates, as described above. The downloadable full report includes uncertainty, additional metrics, and both execution modes.</p>'
};

const server=http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/__render.html'){
    res.setHeader('Content-Type','text/html');
    res.end('<!doctype html><meta charset="utf-8"><script type="module">import mermaid from "/node_modules/mermaid/dist/mermaid.esm.min.mjs"; mermaid.initialize({startOnLoad:false,securityLevel:"strict",theme:"base",themeVariables:{fontFamily:"Arial, sans-serif",fontSize:"14px",primaryColor:"#edf1e8",primaryTextColor:"#252e2b",primaryBorderColor:"#899a89",lineColor:"#7c8e7e",edgeLabelBackground:"#fffdf6",clusterBkg:"#f8f8f2"},flowchart:{htmlLabels:false,curve:"basis",padding:15}});window.mermaid=mermaid;</script>');return;
  }
  const file=path.resolve(root,'.'+decodeURIComponent(url.pathname));
  if(!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  try{res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(await fs.readFile(file));}catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({channel:'chrome',headless:true});
let content=source;
try{
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/__render.html`);
  await page.waitForFunction(()=>!!window.mermaid);
  let n=0;
  for(const match of [...content.matchAll(/~~~mermaid\n([\s\S]*?)\n~~~/g)]){
    const id='diagram-'+n++;
    const svg=await page.evaluate(async({code,id})=>(await window.mermaid.render(id,code)).svg,{code:match[1],id});
    await fs.writeFile(path.join(assetRoot,id+'.svg'),svg);
    content=content.replace(match[0],`<figure class="diagram" aria-label="${['Retrieval and verification pipeline','Fused window encoder training','Benchmark judging and evaluation'][n-1]}">${svg}</figure>`);
  }
}finally{await browser.close();server.close();}
for(const match of [...content.matchAll(/<!-- component: ([\w-]+) -->/g)]){
  if(!components[match[1]])throw new Error('Unknown component '+match[1]);
  content=content.replace(match[0],await components[match[1]]());
}
const chunks=content.split(/(?=^## )/m);
const renderer=new marked.Renderer();
renderer.table=function(token){return '<div class="table-wrap">'+marked.Renderer.prototype.table.call(this,token)+'</div>';};
marked.use({renderer});
const sectionNames=['question','approaches','learning','benchmark','results','generalization','behavior','examples','receipts','uncertainty','choices','next'];
let body=`<header class="hero" id="top">${marked.parse(chunks[0])}</header>`;
for(let i=1;i<chunks.length;i++){
  const html=marked.parse(chunks[i]).replace(/<h2>(\d+) \/ (.*?)<\/h2>/,(_,n,title)=>`<h2><span class="num">${n} / FIELD NOTES</span>${title}</h2>`);
  body+=`<section class="chapter" id="${sectionNames[i-1]}">${html}</section>`;
}
const css=await fs.readFile(path.join(root,'style.css'),'utf8');
const js=await fs.readFile(path.join(root,'client.js'),'utf8');
const embedded=JSON.stringify(data).replaceAll('<','\\u003c');
const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="A measured comparison of embeddings, executable queries, and evidence over real robot recordings."><title>When search needs evidence · Fleet Search</title><style>${css}</style></head><body><div class="topbar"><a class="brand" href="#top"><i></i>Fleet Search</a><nav class="nav" aria-label="Presentation sections"><a href="#approaches">Approaches</a><a href="#benchmark">Benchmark</a><a href="#results">Results</a><a href="#behavior">Behavior</a><a href="#examples">Examples</a><a href="#receipts">Evidence</a></nav><div class="edition">SCAND / v9</div><button id="print" class="print-button" type="button" aria-label="Print this presentation">Print</button><div id="progress"></div></div><main>${body}<footer class="footer">FLEET SEARCH · SCAND SUBSET · SEPTEMBER 2026<br>Frozen eval v9 checkpoint. All media, diagrams, charts, and interactions are embedded.</footer></main><script id="benchmark-data" type="application/json">${embedded}</script><script>${js}</script></body></html>`;
const output=path.join(root,'fleet-search.html');
await fs.writeFile(output,html);
console.log(`Built ${output} (${(Buffer.byteLength(html)/1024/1024).toFixed(2)} MiB)`);
