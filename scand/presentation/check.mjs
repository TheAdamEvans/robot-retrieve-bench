import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {chromium} from 'playwright';
const root=path.dirname(fileURLToPath(import.meta.url));
await fs.mkdir(path.join(root,'.cache'),{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,offline:true,reducedMotion:'reduce'});
 const page=await context.newPage();
 const errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
 await page.goto(pathToFileURL(path.join(root,'fleet-search.html')).href);
 await page.waitForSelector('#metric-table tr');
 assert.equal(await page.locator('.chapter').count(),10);
 assert.equal(await page.locator('.diagram svg').count(),3);
 assert.equal(await page.locator('#metric-table tr').count(),7);
 assert.equal(await page.locator('.plot:visible').count(),1);
 assert.equal(await page.locator('video').count(),3);
 const links=await page.locator('a[href^="#"]').evaluateAll(xs=>xs.map(x=>x.getAttribute('href').slice(1)).filter(id=>!document.getElementById(id)));
 assert.deepEqual(links,[],'Broken section links');
 for(const slice of ['compose_test','demo5_para','demo5'])for(const metric of ['ndcg10','roc_pen'])for(const mode of ['fresh','cached']){
  await page.selectOption('#slice',slice);await page.selectOption('#metric',metric);await page.selectOption('#latency',mode);
  assert.equal(await page.locator('.plot:visible').getAttribute('data-plot'),`${slice}-${metric}-${mode}`);
  assert.equal(await page.locator('#metric-table tr').count(),7);
 }
 await page.selectOption('#slice','compose_test');await page.selectOption('#metric','ndcg10');await page.selectOption('#latency','fresh');
 await page.selectOption('#budget','100');
 assert.match(await page.locator('#budget-result').innerText(),/^Fused has/);
 await page.selectOption('#budget','10000');
 assert.match(await page.locator('#budget-result').innerText(),/^Fused \+ verify has/);
 const hasRemoteAsset=await page.evaluate(()=>[...document.querySelectorAll('img,source,script[src],link[rel="stylesheet"]')].some(x=>x.src&&!x.src.startsWith('data:')));
 assert.equal(hasRemoteAsset,false);
 await page.evaluate(()=>document.querySelectorAll('img').forEach(i=>i.loading='eager'));
 await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0));
 for(let i=0;i<3;i++){
  const v=page.locator('video').nth(i);
  await v.evaluate(v=>{v.preload='auto';v.load();});
  await page.waitForFunction(i=>document.querySelectorAll('video')[i].readyState>=2,i);
  await v.evaluate(async v=>{v.currentTime=.5;await v.play();});
  await page.waitForFunction(i=>document.querySelectorAll('video')[i].currentTime>.55,i);
  await v.evaluate(v=>v.pause());
 }
 await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(root,'.cache','desktop-top.png')});
 for(const [id,name] of [['results','desktop-results'],['receipts','desktop-receipt']]){
  await page.locator('#'+id).scrollIntoViewIfNeeded();
  await page.evaluate(id=>scrollTo(0,document.getElementById(id).offsetTop-90),id);
  await page.screenshot({path:path.join(root,'.cache',name+'.png')});
 }
 await page.setViewportSize({width:390,height:844});
 await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(root,'.cache','mobile-top.png')});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Horizontal page overflow on mobile');
 await page.locator('#results').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(root,'.cache','mobile-results.png')});
 assert.deepEqual(errors,[],'Browser errors');assert.deepEqual(requests,[],'Network requests');
 console.log('PASS: offline file:// load, 12 explorer modes, budget controls, section links, all images, three playable videos, desktop/mobile layouts; zero network requests.');
 await context.close();
}finally{await browser.close();}
