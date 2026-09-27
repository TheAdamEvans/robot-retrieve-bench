const data = JSON.parse(document.getElementById('benchmark-data').textContent);
const $ = id => document.getElementById(id);
const selectedConfigs = ['TAGS','EMBED','FUSED','PROGRAM_LUNA','HYBRID_LUNA','FUSED_V_LUNA','PROGRAM_ORACLE'];
const fmtTime = x => x < 1 ? '<1 ms' : x < 1000 ? `${Math.round(x)} ms` : `${(x/1000).toFixed(2)} s`;
const names = data.names;
function updateExplorer(){
  const split=$('slice').value, metric=$('metric').value, mode=$('latency').value, budget=+$('budget').value;
  const latencyKey=mode==='fresh'?'wall_p50':'wall_cached_p50';
  const table=data.report.tables[split];
  const rows=selectedConfigs.map(c=>table.find(r=>r.config===c)).filter(Boolean);  // the challenge set has no oracle program
  document.querySelectorAll('.plot').forEach(p=>p.hidden=p.dataset.plot!==`${split}-${metric}-${mode}`);
  $('metric-title').textContent=metric==='ndcg10'?'nDCG@10':'ROC-AUC';
  $('metric-table').innerHTML=rows.map(r=>{
    const m=r[metric],oracle=r.config==='PROGRAM_ORACLE';
    const usage=mode==='cached'&&r.config.endsWith('LUNA')?0:r.tokens.mean;
    return `<tr class="${oracle?'oracle-row':''} ${!oracle&&r[latencyKey]>budget?'excluded':''}"><td><span class="swatch" style="background:${data.colors[r.config]}"></span>${names[r.config]}${oracle?' †':''}</td><td>${m.mean.toFixed(2)} ${m.ci?`<span class="ci-note">[${m.ci.map(v=>v.toFixed(2)).join('–')}]</span>`:''}</td><td>${fmtTime(r[latencyKey])}</td><td>${Math.round(usage).toLocaleString()}</td><td>${(100*r.judged50.mean).toFixed(0)}%</td></tr>`;
  }).join('');
  const eligible=rows.filter(r=>r.config!=='PROGRAM_ORACLE'&&r[latencyKey]<=budget).sort((a,b)=>b[metric].mean-a[metric].mean);
  $('budget-result').innerHTML=eligible.length?`<strong>${names[eligible[0].config]}</strong> has the highest point estimate within this median-latency budget.<small>Exploration aid; differences have uncertainty. Oracle excluded. A median is not a tail-latency guarantee.</small>`:'No measured serving approach fits this budget.';
  const n=rows[0][metric].n_groups;
  $('chart-note').textContent=`Bars show point estimates; whiskers show 95% bootstrap intervals over ${n} intent groups. ${mode==='fresh'?'Fresh-program latency uses original measured generation time plus replay execution.':'Cached-program latency measures replay execution; no new generation tokens.'} † Oracle uses a supplied program.`;
}
['slice','metric','latency','budget'].forEach(id=>$(id).addEventListener('change',updateExplorer));
updateExplorer();
const progress=$('progress');
function updateProgress(){const range=document.documentElement.scrollHeight-innerHeight;progress.style.width=`${range>0?100*scrollY/range:0}%`;}
addEventListener('scroll',updateProgress,{passive:true});addEventListener('resize',updateProgress);updateProgress();
document.querySelectorAll('video').forEach(video=>video.addEventListener('play',()=>document.querySelectorAll('video').forEach(v=>{if(v!==video)v.pause();})));
$('print').addEventListener('click',()=>print());
