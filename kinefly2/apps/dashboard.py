"""Live browser view of the fly state, in the style of a chart recorder.

Subscribes to the ZMQ state stream published by run_rig and serves one
self-contained HTML page (no external libraries, so it works on an offline rig
network). It runs as a separate process, so it adds no load to the tracker, and
can be opened from any machine that can reach the rig.

    python -m kinefly2.apps.dashboard                         # tracker on this machine
    python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
    then open  http://<this-machine>:8000

Each trace holds a FIXED y range, so the picture stays still and a value now is
comparable with the same value a minute ago; autoscaling on every redraw makes
small changes invisible by normalising them away. Per-trace controls set the
range, and the layout is remembered between reloads.

Channel names follow the tracker: each camera measures one rotation exactly
(its in-plane angle) and, for a frontal view, two more through a lever arm.
Those come from the part's `extra`, so the labels match what the numbers are
regardless of camera mounting -- reading the raw image-axis displacement would
mislabel them on any camera whose axis is not aligned with the frame.
"""
import argparse
import collections
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq

MAX_HISTORY_S = 65.0        # server buffer; the page picks a window within this
PARTS = ('left', 'right', 'head', 'abdomen')


def clean(x):
    """JSON has no NaN or Infinity. Python's json emits bare `NaN`, which
    JSON.parse rejects outright -- one uncalibrated channel would kill the
    whole stream -- so non-finite values become null and are drawn as gaps."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


class StateBuffer:
    def __init__(self, zmq_addr, history_s=MAX_HISTORY_S):
        self.samples = collections.deque()
        self.lock = threading.Lock()
        self.history_s = history_s
        self.n = 0
        self.t_last = None
        self.zmq_addr = zmq_addr
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def row_from_state(state):
        """Flatten one published state into the channels the page plots."""
        row = {'t': state['t_host'], 'seq': state['seq'],
               'stim': state.get('stim')}
        for cam, cs in (state.get('cameras') or {}).items():
            parts = cs.get('parts') or {}
            for part in PARTS:
                a = ((parts.get(part) or {}).get('angles') or [None])[0]
                row['%s/%s' % (cam, part)] = None if a is None else math.degrees(a)
            head = parts.get('head') or {}
            extra = head.get('extra') or {}
            # The anatomical channels, named by the tracker itself. Exactly one
            # is the exact in-plane angle; a frontal camera fills the other two
            # once its lever arm is calibrated, a ventral one never does.
            for name in ('roll', 'pitch', 'yaw'):
                v = extra.get(name)
                row['%s/head_%s' % (cam, name)] = None if v is None else math.degrees(v)
            # Uncalibrated fallbacks, in pixels.
            for name in ('d_axial', 'd_lateral'):
                row['%s/head_%s' % (cam, name)] = extra.get(name)
            row['%s/head_q' % cam] = head.get('intensity')
            aux = parts.get('aux') or {}
            row['%s/wbf' % cam] = aux.get('freq')
        return clean(row)

    def _run(self):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.connect(self.zmq_addr)
        sock.setsockopt_string(zmq.SUBSCRIBE, '')
        sock.setsockopt(zmq.RCVTIMEO, 500)
        while True:
            try:
                state = json.loads(sock.recv_string())
            except zmq.Again:
                continue
            except (ValueError, zmq.ZMQError):
                continue
            row = self.row_from_state(state)
            with self.lock:
                self.samples.append(row)
                self.n += 1
                self.t_last = time.time()
                while self.samples and row['t'] - self.samples[0]['t'] > self.history_s:
                    self.samples.popleft()

    def since(self, t):
        with self.lock:
            return [s for s in self.samples if s['t'] > t]

    def all(self):
        with self.lock:
            return list(self.samples)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Kinefly2 live</title>
<style>
:root{--bg:#111;--panel:#181818;--line:#333;--fg:#ddd;--dim:#888}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:7px 12px;border-bottom:1px solid var(--line);
 position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600}
#status{color:#8c8}
button,select{background:#222;color:var(--fg);border:1px solid #444;border-radius:4px;
 padding:2px 7px;font:12px system-ui;cursor:pointer}
button:hover{background:#2c2c2c}
button.on{background:#3a5;border-color:#4c6;color:#fff}
.panel{margin:3px 10px;border:1px solid var(--line);background:var(--panel);border-radius:4px}
.panel.drag{opacity:.4}
.hd{display:flex;align-items:center;gap:8px;padding:3px 7px;font-size:12px;border-bottom:1px solid #262626}
.hd .name{font-weight:600;cursor:grab;user-select:none}
.hd .cur{margin-left:auto;font-variant-numeric:tabular-nums}
.hd .ctl{display:flex;gap:3px}
.hd input{width:58px;background:#1e1e1e;color:var(--fg);border:1px solid #444;border-radius:3px;
 padding:1px 3px;font:11px ui-monospace,monospace;text-align:right}
canvas{width:100%;height:104px;display:block;cursor:crosshair}
#hidden{padding:4px 12px 10px;color:var(--dim);font-size:12px}
#hidden button{margin:2px 3px 0 0;opacity:.75}
</style></head><body>
<header>
  <h1>Kinefly2 live</h1><span id="status">connecting…</span>
  <label>window <select id="win">
    <option value="5">5 s</option><option value="10">10 s</option>
    <option value="30" selected>30 s</option><option value="60">60 s</option>
  </select></label>
  <button id="pause">Pause</button>
  <button id="fitall">Fit all</button>
  <button id="reset">Reset layout</button>
  <span id="cursor" style="color:var(--dim)"></span>
</header>
<div id="plots"></div>
<div id="hidden"></div>
<script>
"use strict";
// Channel registry. `src` is the row key; `alt` is the uncalibrated fallback
// used when the calibrated one is absent, so a front camera reads in degrees
// once lever_px is set and in pixels before that.
const CHANNELS=[
 {id:'wsa',   title:'Wing stroke amplitude (bottom)', unit:'deg',
  keys:[{k:'bottom/left',c:'#e55',lbl:'L'},{k:'bottom/right',c:'#5be',lbl:'R'}], def:[0,180]},
 {id:'lmr',   title:'L − R  (bottom)', unit:'deg', keys:[{k:'L-R',c:'#fd5',lbl:'L−R'}], def:[-60,60]},
 {id:'byaw',  title:'Head yaw (bottom, exact)', unit:'deg',
  keys:[{k:'bottom/head_yaw',c:'#8f8',lbl:'yaw'}], def:[-30,30]},
 {id:'froll', title:'Head roll (front, exact)', unit:'deg',
  keys:[{k:'front/head_roll',c:'#c8f',lbl:'roll'}], def:[-30,30]},
 {id:'fpitch',title:'Head pitch (front)', unit:'deg',
  keys:[{k:'front/head_pitch',c:'#8cf',lbl:'pitch',alt:'front/head_d_axial',altUnit:'px'}], def:[-20,20]},
 {id:'fyaw',  title:'Head yaw (front)', unit:'deg',
  keys:[{k:'front/head_yaw',c:'#5c9',lbl:'yaw',alt:'front/head_d_lateral',altUnit:'px'}], def:[-30,30]},
 {id:'stim',  title:'Stimulus code', unit:'V', keys:[{k:'stim',c:'#fa6',lbl:'stim'}], def:[-1,11], step:true},
 {id:'wbf',   title:'Wingbeat frequency (bottom)', unit:'Hz', keys:[{k:'bottom/wbf',c:'#f96',lbl:'wbf'}], def:[150,250]},
 {id:'q',     title:'Match quality', unit:'',
  keys:[{k:'bottom/head_q',c:'#7a7',lbl:'bottom'},{k:'front/head_q',c:'#a7a',lbl:'front'}], def:[0,1]},
];
const LS='kinefly2.dash.v1';
let cfg=load();
function load(){ try{const o=JSON.parse(localStorage.getItem(LS)); if(o&&o.panels) return o;}catch(e){} return blank(); }
function blank(){ return {win:30, panels:CHANNELS.map(c=>({id:c.id,lo:c.def[0],hi:c.def[1],show:null}))}; }
function save(){ try{localStorage.setItem(LS,JSON.stringify(cfg));}catch(e){} }
function specOf(id){ return CHANNELS.find(c=>c.id===id); }
function panelOf(id){ return cfg.panels.find(p=>p.id===id); }

const data=[]; let last=null, n=0, rate={t:performance.now(),n:0};
let paused=false, frozen=null, cursorT=null;
const root=document.getElementById('plots'), hiddenBox=document.getElementById('hidden');

function val(row,k){
  if(k==='L-R'){const l=row['bottom/left'],r=row['bottom/right'];return (l==null||r==null)?null:l-r;}
  const v=row[k]; return v==null?null:v;
}
// A channel is "live" if anything finite arrived for it recently.
function hasData(spec){
  for(const kk of spec.keys){
    for(let i=data.length-1;i>=0&&i>data.length-400;i--){
      if(val(data[i],kk.k)!=null) return {key:kk.k,unit:spec.unit};
      if(kk.alt&&val(data[i],kk.alt)!=null) return {key:kk.alt,unit:kk.altUnit};
    }
  }
  return null;
}
function activeKey(kk){
  for(let i=data.length-1;i>=0&&i>data.length-400;i--){
    if(val(data[i],kk.k)!=null) return {k:kk.k,unit:null};
    if(kk.alt&&val(data[i],kk.alt)!=null) return {k:kk.alt,unit:kk.altUnit};
  }
  return {k:kk.k,unit:null};
}

function build(){
  root.innerHTML=''; hiddenBox.innerHTML='';
  for(const p of cfg.panels){
    const spec=specOf(p.id); if(!spec) continue;
    if(p.show===false){ const b=document.createElement('button');
      b.textContent='+ '+spec.title; b.onclick=()=>{p.show=true;save();build();}; hiddenBox.appendChild(b); continue; }
    const d=document.createElement('div'); d.className='panel'; d.dataset.id=p.id; d.draggable=false;
    d.innerHTML=`<div class="hd">
      <span class="name" draggable="true">${spec.title}</span>
      <span class="ctl">
        <button title="Fit this trace to the visible window">fit</button>
        <button title="Zoom in">+</button><button title="Zoom out">−</button>
        <button title="Pan up">▲</button><button title="Pan down">▼</button>
        <input class="lo" value="${p.lo}"><input class="hi" value="${p.hi}">
        <button title="Hide this trace">×</button>
      </span>
      <span class="cur"></span></div><canvas></canvas>`;
    const btns=d.querySelectorAll('.hd .ctl button');
    btns[0].onclick=()=>{autofit(p);save();};
    btns[1].onclick=()=>{zoom(p,1/1.5);save();};
    btns[2].onclick=()=>{zoom(p,1.5);save();};
    btns[3].onclick=()=>{pan(p,+0.2);save();};
    btns[4].onclick=()=>{pan(p,-0.2);save();};
    btns[5].onclick=()=>{p.show=false;save();build();};
    const lo=d.querySelector('.lo'), hi=d.querySelector('.hi');
    for(const [el,f] of [[lo,'lo'],[hi,'hi']])
      el.onchange=()=>{const v=parseFloat(el.value); if(isFinite(v)){p[f]=v;save();}};
    const nm=d.querySelector('.name');
    nm.ondragstart=e=>{e.dataTransfer.setData('text/plain',p.id);d.classList.add('drag');};
    nm.ondragend=()=>d.classList.remove('drag');
    d.ondragover=e=>e.preventDefault();
    d.ondrop=e=>{e.preventDefault(); const from=e.dataTransfer.getData('text/plain');
      if(!from||from===p.id)return; const a=cfg.panels.findIndex(x=>x.id===from), b=cfg.panels.findIndex(x=>x.id===p.id);
      const [m]=cfg.panels.splice(a,1); cfg.panels.splice(b,0,m); save(); build();};
    const cv=d.querySelector('canvas');
    cv.onmousemove=e=>{const r=cv.getBoundingClientRect(); const W=r.width;
      const now=(frozen||last)?(frozen||last).t:0; cursorT=now-cfg.win*(1-(e.clientX-r.left)/W); draw();};
    cv.onmouseleave=()=>{cursorT=null;draw();};
    root.appendChild(d); p.el=d;
  }
}
function zoom(p,f){const c=(p.lo+p.hi)/2,h=(p.hi-p.lo)/2*f;p.lo=c-h;p.hi=c+h;sync(p);}
function pan(p,f){const s=(p.hi-p.lo)*f;p.lo+=s;p.hi+=s;sync(p);}
function sync(p){ if(!p.el)return; p.el.querySelector('.lo').value=p.lo.toFixed(2); p.el.querySelector('.hi').value=p.hi.toFixed(2); }
function autofit(p){
  const spec=specOf(p.id); let lo=Infinity,hi=-Infinity;
  const now=(frozen||last)?(frozen||last).t:0;
  for(const kk of spec.keys){ const key=activeKey(kk).k;
    for(const r of data){ if(now-r.t>cfg.win)continue; const v=val(r,key);
      if(v!=null&&isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);} } }
  if(!isFinite(lo))return;
  const pad=(hi-lo)*0.12||Math.abs(hi)*0.1||1; p.lo=lo-pad; p.hi=hi+pad; sync(p);
}
function nice(v){ const a=Math.abs(v); return a>=100?v.toFixed(0):a>=10?v.toFixed(1):v.toFixed(2); }

function draw(){
  const ref=frozen||last; const now=ref?ref.t:0;
  for(const p of cfg.panels){
    if(p.show===false||!p.el)continue;
    const spec=specOf(p.id), cv=p.el.querySelector('canvas');
    const W=cv.width=cv.clientWidth*devicePixelRatio, H=cv.height=cv.clientHeight*devicePixelRatio;
    const g=cv.getContext('2d'); g.clearRect(0,0,W,H);
    const Y=v=>H*(1-(v-p.lo)/(p.hi-p.lo));
    // grid: 4 horizontal divisions + a second-marked time axis
    g.strokeStyle='#282828'; g.lineWidth=1; g.beginPath();
    for(let i=1;i<4;i++){const y=H*i/4;g.moveTo(0,y);g.lineTo(W,y);}
    const step=cfg.win<=10?1:cfg.win<=30?5:10;
    for(let s=0;s<=cfg.win;s+=step){const x=W*(1-s/cfg.win);g.moveTo(x,0);g.lineTo(x,H);}
    g.stroke();
    if(p.lo<0&&p.hi>0){g.strokeStyle='#444';g.beginPath();g.moveTo(0,Y(0));g.lineTo(W,Y(0));g.stroke();}
    let unitShown=spec.unit;
    spec.keys.forEach(kk=>{
      const act=activeKey(kk); if(act.unit)unitShown=act.unit;
      // Min/max decimation: one vertical segment per pixel column. A 60 s
      // window at 90 Hz is 5400 points; drawing them all both aliases and drags.
      const cols=new Array(Math.ceil(W)).fill(null);
      for(const r of data){
        const dt=now-r.t; if(dt<0||dt>cfg.win)continue;
        const v=val(r,act.k); if(v==null||!isFinite(v))continue;
        const x=Math.floor(W*(1-dt/cfg.win)); if(x<0||x>=cols.length)continue;
        const c=cols[x]; if(c==null)cols[x]={lo:v,hi:v,last:v}; else {c.lo=Math.min(c.lo,v);c.hi=Math.max(c.hi,v);c.last=v;}
      }
      g.strokeStyle=kk.c; g.lineWidth=1.4*devicePixelRatio; g.beginPath();
      let pen=false, prev=null;
      for(let x=0;x<cols.length;x++){
        const c=cols[x]; if(c==null){pen=false;continue;}
        if(spec.step&&pen&&prev!=null){ g.lineTo(x,Y(prev)); }  // hold: square edges, never across a gap
        if(!pen){g.moveTo(x,Y(c.last));pen=true;} else {g.lineTo(x,Y(c.hi)); if(c.hi!==c.lo)g.lineTo(x,Y(c.lo)); g.lineTo(x,Y(c.last));}
        prev=c.last;
      }
      g.stroke();
    });
    // axis labels
    g.fillStyle='#777'; g.font=`${10*devicePixelRatio}px ui-monospace,monospace`;
    g.fillText(nice(p.hi),3,11*devicePixelRatio);
    g.fillText(nice(p.lo),3,H-3);
    g.fillText(`-${cfg.win}s`,3,H/2+4*devicePixelRatio);
    g.textAlign='right'; g.fillText('0',W-3,H/2+4*devicePixelRatio); g.textAlign='left';
    if(cursorT!=null){ const x=W*(1-(now-cursorT)/cfg.win);
      g.strokeStyle='#999';g.setLineDash([3,3]);g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();g.setLineDash([]); }
    // current values
    let at=ref;
    if(cursorT!=null){ let best=null,bd=1e9; for(const r of data){const d=Math.abs(r.t-cursorT); if(d<bd){bd=d;best=r;}} at=best||ref; }
    p.el.querySelector('.cur').innerHTML = spec.keys.map(kk=>{
      const act=activeKey(kk); const v=at?val(at,act.k):null;
      return `<span style="color:${kk.c}">${kk.lbl} ${v==null?'–':nice(v)}</span>`;
    }).join(' &nbsp; ')+` <span style="color:var(--dim)">${unitShown}</span>`;
  }
}

document.getElementById('win').onchange=e=>{cfg.win=+e.target.value;save();draw();};
document.getElementById('pause').onclick=e=>{
  paused=!paused; frozen=paused?last:null; e.target.classList.toggle('on',paused);
  e.target.textContent=paused?'Live':'Pause'; draw();};
document.getElementById('fitall').onclick=()=>{cfg.panels.forEach(p=>{if(p.show!==false)autofit(p);});save();draw();};
document.getElementById('reset').onclick=()=>{cfg=blank();save();build();draw();};
document.getElementById('win').value=String(cfg.win);

const es=new EventSource('/stream');
es.onmessage=e=>{
  let rows; try{rows=JSON.parse(e.data);}catch(err){document.getElementById('status').textContent='bad frame';return;}
  for(const r of rows){data.push(r);n++;} if(rows.length)last=data[data.length-1];
  const keep=Math.max(cfg.win,60)+5;
  while(data.length&&last.t-data[0].t>keep)data.shift();
  const dt=(performance.now()-rate.t)/1000;
  if(dt>2){document.getElementById('status').textContent=`${((n-rate.n)/dt).toFixed(0)} states/s`;rate={t:performance.now(),n:n};}
  // First data in: hide panels that carry nothing, so a one-camera session
  // is not half empty. The user's own choices are never overridden.
  if(!window._autoHid&&data.length>60){ window._autoHid=true;
    let ch=false; for(const p of cfg.panels){ if(p.show===null){ p.show=!!hasData(specOf(p.id)); ch=true; } }
    if(ch){save();build();} }
};
es.onerror=()=>{document.getElementById('status').textContent='disconnected';};
build(); setInterval(()=>{if(!paused)draw();},50);
</script></body></html>"""


def make_handler(buf):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self._send(PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/history':
                self._send(json.dumps(buf.all()).encode(), 'application/json')
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                t = -math.inf
                try:
                    rows = buf.all()
                    if rows:
                        self.wfile.write(('data: %s\n\n' % json.dumps(rows)).encode())
                        t = rows[-1]['t']
                    while True:
                        time.sleep(0.05)
                        rows = buf.since(t)
                        if rows:
                            t = rows[-1]['t']
                            self.wfile.write(('data: %s\n\n' % json.dumps(rows)).encode())
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zmq', default='tcp://localhost:5556', help='run_rig ZMQ publisher address')
    ap.add_argument('--port', type=int, default=8000, help='HTTP port to serve the page on')
    args = ap.parse_args()
    buf = StateBuffer(args.zmq)
    server = ThreadingHTTPServer(('', args.port), make_handler(buf))
    print('Dashboard: http://localhost:%d  (fly state from %s)' % (args.port, args.zmq))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
