// Browser smoke for the live study UI, over WebDriver BiDi (Firefox).
// Chrome headless cannot commit navigations on this machine; launch:
//   firefox --headless --remote-debugging-port 9225 --profile <fresh> --no-remote
// One BiDi session per browser process; restart Firefox between runs.
import fs from 'node:fs/promises';
const base=process.argv[2] || 'http://127.0.0.1:8877';
const out=process.argv[3] || 'native/artifacts/rl/opponent-extension';
const port=process.env.BIDI_PORT || '9225';

const ws=new WebSocket(`ws://127.0.0.1:${port}/session`);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',()=>reject(Error('BiDi port unreachable')),{once:true});});
let id=0;const pending=new Map(),errors=[];
ws.addEventListener('message',event=>{const m=JSON.parse(event.data);
  if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timer);m.type==='error'?p.reject(Error(JSON.stringify(m))):p.resolve(m);}
  else if(m.method==='log.entryAdded'&&m.params.level==='error')errors.push({context:m.params.source?.context,text:m.params.text});});
function call(method,params={}){return new Promise((resolve,reject)=>{const next=++id;const timer=setTimeout(()=>{pending.delete(next);reject(Error(method+' timed out'));},30000);pending.set(next,{resolve,reject,timer});ws.send(JSON.stringify({id:next,method,params}));});}
await call('session.new',{capabilities:{}});
await call('session.subscribe',{events:['log.entryAdded']});
const context=(await call('browsingContext.getTree',{})).result.contexts[0].context;
async function evaluate(expression){
  const r=await call('script.evaluate',{expression:`JSON.stringify((()=>(${expression}))() ?? null)`,target:{context},awaitPromise:true});
  if(r.result.type!=='success')throw Error('evaluate failed: '+JSON.stringify(r.result).slice(0,500));
  return JSON.parse(r.result.result.value);}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(expression){for(let i=0;i<900;i++){if(await evaluate(expression))return;await sleep(200);}throw Error('Wait failed: '+expression);}
async function screenshot(path){const shot=await call('browsingContext.captureScreenshot',{context,origin:'viewport'});await fs.writeFile(path,Buffer.from(shot.result.data,'base64'));}
async function viewport(width,height){await call('browsingContext.setViewport',{context,viewport:{width,height}});}

await viewport(1440,1100);
await call('browsingContext.navigate',{context,url:base+'/',wait:'complete'});
await until("typeof currentStudy!=='undefined' && currentStudy?.name==='opponent-extension'");
const desktop=await evaluate(`({runs:document.querySelectorAll('.study-run').length,wide:document.documentElement.scrollWidth>innerWidth,crossB:document.querySelector('#crossB').textContent,crossMilestone:document.querySelector('#crossplayMilestone').value,expectedCrossB:pct(currentStudy.crossplay[document.querySelector('#crossplayMilestone').value].B_wins/currentStudy.crossplay[document.querySelector('#crossplayMilestone').value].games),latestComplete:Object.entries(currentStudy.crossplay).filter(([,c])=>c.complete).at(-1)?.[0],cells:document.querySelectorAll('#pairingRows td').length,rows:document.querySelectorAll('#milestoneRows tr').length,clips:document.querySelector('#studyClip').options.length,failures:!document.querySelector('#failureSection').hidden,remaining:currentStudy.total_target-currentStudy.total_learner_steps})`);
if(desktop.runs!==6 || desktop.rows!==6 || desktop.cells!==9 || desktop.wide || desktop.crossB!==desktop.expectedCrossB || desktop.crossMilestone!==desktop.latestComplete || !desktop.failures)throw Error('Desktop data mismatch '+JSON.stringify(desktop));
await evaluate("document.querySelector('.study-details').open=true,document.querySelector('#detailMilestone').value='1000000',document.querySelector('#detailSuite').value='transfer',document.querySelector('#detailSuite').dispatchEvent(new Event('change')),1");
if(await evaluate("document.querySelectorAll('#detailRows tr').length")!==12)throw Error('Opponent breakdown missing');
await evaluate("document.querySelector('.study-details').open=false,document.querySelector('#studyVideo').muted=true,document.querySelector('#studyVideo').play(),1");
await until("document.querySelector('#studyVideo').readyState>=2 && document.querySelector('#studyVideo').currentTime>0.1");
await evaluate("document.querySelector('#studyVideo').currentTime=7");
await until("!document.querySelector('#studyVideo').seeking && document.querySelector('#studyVideo').currentTime>7.05");
await evaluate("document.querySelector('#studyVideo').pause(),1");
await screenshot(out+'/dashboard-desktop.png');
await until("document.querySelector('#matchPicker').options.length>100");
await evaluate("document.querySelector('#matchCheckpoint').value='1000000',document.querySelector('#matchCheckpoint').dispatchEvent(new Event('change')),document.querySelector('#matchWatch').click(),1");
await until("!document.querySelector('#matchStage').hidden");
await evaluate("document.querySelector('#matchPlay').click(),1");
await until("Number(document.querySelector('#matchSeek').value)>2");
await evaluate("document.querySelector('#matchPlay').click(),document.querySelector('#matchSeek').value=100,document.querySelector('#matchSeek').dispatchEvent(new Event('input')),document.querySelector('#matchSpeed').value='4',document.querySelector('#matchPlay').click(),1");
await until("Number(document.querySelector('#matchSeek').value)>110");
await evaluate("document.querySelector('#matchPlay').click(),document.querySelector('#matchSeek').value=document.querySelector('#matchSeek').max,document.querySelector('#matchSeek').dispatchEvent(new Event('input')),1");
const replay=await evaluate("({status:document.querySelector('#matchStatus').textContent,identity:document.querySelector('#matchIdentity').textContent,time:document.querySelector('#matchTime').textContent,hud:document.querySelector('#matchHud').textContent,saved:document.querySelector('#matchPicker').options.length})");
await evaluate("document.querySelector('#matchNext').click(),1");
await until("!document.querySelector('#matchStage').hidden");
await evaluate("document.querySelector('#matchStage').scrollIntoView({block:'center'}),1");
await screenshot(out+'/dashboard-replay.png');
await viewport(390,844);await sleep(300);
const mobile=await evaluate("({wide:document.documentElement.scrollWidth>innerWidth,runs:document.querySelectorAll('.study-run').length,videoWidth:document.querySelector('#studyVideo').getBoundingClientRect().width})");
if(mobile.wide || mobile.videoWidth>390)throw Error('Mobile overflow '+JSON.stringify(mobile));
await screenshot(out+'/dashboard-mobile.png');
await evaluate("document.querySelector('#matchStage').scrollIntoView({block:'center'}),1");
await screenshot(out+'/dashboard-replay-mobile.png');
await viewport(1440,1100);
await evaluate("document.querySelector('#studyPicker').value='opponent-league',document.querySelector('#studyPicker').dispatchEvent(new Event('change')),1");
await until("currentStudy?.name==='opponent-league' && document.querySelectorAll('.study-run').length===9");
if(!await evaluate("!document.querySelector('#retentionSection').hidden && document.querySelector('#crossplaySection').hidden"))throw Error('Previous study did not render');
await call('browsingContext.navigate',{context,url:base+'/?series=dap-multimap-001',wait:'complete'});
await until("document.querySelector('#run') && document.querySelector('#connection').textContent.startsWith('Live') && !document.querySelector('#play').disabled");
const legacy=await evaluate("({steps:document.querySelector('#steps').textContent,replay:!document.querySelector('#play').disabled,runs:document.querySelector('#run').options.length,wide:document.documentElement.scrollWidth>innerWidth})");
if(legacy.steps==='—' || !legacy.replay || legacy.wide)throw Error('Legacy view failed '+JSON.stringify(legacy));
const pageErrors=errors.filter(e=>e.context===context).map(e=>e.text);
if(pageErrors.length)throw Error('Browser exceptions '+JSON.stringify(pageErrors));
const report={desktop,mobile,legacy,replay,full_replay_play_seek_speed_next:true,video_playback_and_seeking:true,study_switching:true,console_errors:pageErrors};await fs.writeFile(out+'/dashboard-check.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));ws.close();process.exit(0);
