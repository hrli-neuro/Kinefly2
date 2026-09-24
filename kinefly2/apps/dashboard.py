"""Live browser view of the fly state, and a viewer for saved recordings.

Subscribes to the ZMQ state stream published by run_rig and serves one
self-contained HTML page (no external libraries, so it works on an offline rig
network). It runs as a separate process, so it adds no load to the tracker, and
can be opened from any machine that can reach the rig.

    python -m kinefly2.apps.dashboard                         # tracker on this machine
    python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
    python -m kinefly2.apps.dashboard --file <run>_flystate.h5   # a saved recording
    then open  http://<this-machine>:8000

`--file` reads a recording back through the SAME page: the loader turns HDF5
columns into exactly the row dicts the live path produces, so every control --
scrolling, the minimap, markers, the cursor -- works identically on a saved
file. Saved files add event markers, which the live stream does not carry.

TIME. The x-axis is the PRIMARY CAMERA'S FRAME CLOCK (`t_camera`), not the
host clock: `frame_id / fps` for a video file and the device clock for a real
camera. That makes a 10 s window mean 10 s of experiment either way. Plotting
the host clock instead (what this did originally) silently rescales replay --
tracking a 60 fps file as fast as the machine manages ran 6.6x real time, so a
"10 s" window was showing 66 s of video. `t_host` is kept on every row for
buffer trimming and for placing events, because a looping video resets the
frame clock while the host clock never goes backwards.

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
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..scope.data import PARTS, axis_time, clean, load_recording, row_from_state  # noqa: F401

MAX_HISTORY_S = 300.0       # server buffer: one recording's worth (experiments run < 3 min)
MAX_ROWS = 120000           # hard cap, so a reset frame clock cannot grow the buffer forever


class StateBuffer:
    """The live source: rows arriving over ZMQ, trimmed to a time window."""

    def __init__(self, zmq_addr, history_s=MAX_HISTORY_S):
        self.samples = collections.deque()
        self.lock = threading.Lock()
        self.history_s = history_s
        self.n = 0
        self.t_last = None
        self.zmq_addr = zmq_addr
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # Flattening a published state is shared with the Scope (scope/data.py);
    # kept reachable here, where the tests and older scripts look for it.
    row_from_state = staticmethod(row_from_state)

    def _run(self):
        import zmq

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
                self.trim()

    def trim(self):
        """Drop what has aged out. Caller holds the lock.

        Trimmed on the HOST clock, never the display axis: a looping video
        resets its frame clock, so a span measured on `t` would go negative and
        never trim again. MAX_ROWS is the backstop for any clock that misbehaves
        in a way this does not anticipate."""
        while self.samples and (
                self.samples[-1]['t_host'] - self.samples[0]['t_host'] > self.history_s
                or len(self.samples) > MAX_ROWS):
            self.samples.popleft()

    def since(self, t_host):
        with self.lock:
            return [s for s in self.samples if s['t_host'] > t_host]

    def all(self):
        with self.lock:
            return list(self.samples)

    def meta(self):
        with self.lock:
            last = self.samples[-1] if self.samples else None
        return {'mode': 'live', 'name': None, 'history_s': self.history_s,
                'events': [], 'rec': (last or {}).get('rec'), 'n': self.n}


def rows_from_h5(path):
    """Load a saved recording into the same rows the live path publishes.

    One data model is the whole point: the page cannot tell a file from a live
    stream, so scrolling, markers, the minimap and the cursor need no second
    implementation. The reading itself (channel names, degrees, the display
    axis, stale rows, events) is scope/data.py's, shared with the Scope."""
    rec = load_recording(path)
    n = len(rec.t)
    if n == 0:
        raise SystemExit('%s has no rows -- was the recording stopped before any '
                         'frames arrived?' % path)

    def tolist(a):
        return [None if v != v else v for v in a.tolist()]

    channels = {k: tolist(v) for k, v in rec.columns.items()}
    axis_l, host_l, seq_l = rec.t.tolist(), rec.t_host.tolist(), rec.seq.tolist()
    rows = []
    for i in range(n):
        row = {'t': axis_l[i], 't_host': host_l[i], 'seq': int(seq_l[i]),
               'stale': rec.stale[i], 'rec': rec.name}
        for key, vals in channels.items():
            row[key] = vals[i]
        rows.append(row)
    i = rec.info
    info = {'mode': 'file', 'name': rec.name, 'primary': i['primary'],
            'primary_known': i['primary_known'], 'cameras': i['cameras'],
            'stale_after_s': i['stale_after_s'], 'events': rec.events, 'n': n,
            'span': float(axis_l[-1] - axis_l[0]),
            'code_version': i.get('code_version'),
            'conventions': i.get('conventions'),
            'lever_px': i.get('lever_px')}
    return clean(rows), clean(info)


class FileBuffer:
    """A saved recording, presented with the same interface as StateBuffer."""

    def __init__(self, path):
        self.path = path
        self.rows, self.info = rows_from_h5(path)

    def all(self):
        return self.rows

    def since(self, t_host):
        return []               # A file never grows.

    def meta(self):
        return dict(self.info, history_s=None)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Kinefly2 live</title>
<style>
:root{--bg:#111;--panel:#181818;--line:#333;--fg:#ddd;--dim:#888}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:7px 12px;border-bottom:1px solid var(--line);
 position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600}
#status{color:#8c8}
#stale{color:#fff;background:#a22;padding:2px 7px;border-radius:3px;margin-left:8px;display:none}
#rec{color:#fff;background:#c33;padding:2px 7px;border-radius:3px;display:none}
#when{color:var(--dim);font-variant-numeric:tabular-nums}
button,select{background:#222;color:var(--fg);border:1px solid #444;border-radius:4px;
 padding:2px 7px;font:12px system-ui;cursor:pointer}
button:hover{background:#2c2c2c}
button.on{background:#3a5;border-color:#4c6;color:#fff}
#mapwrap{margin:5px 10px 0;border:1px solid var(--line);background:var(--panel);border-radius:4px}
#map{width:100%;height:38px;display:block;cursor:ew-resize}
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
  <h1 id="title">Kinefly2 live</h1><span id="status">connecting…</span>
  <span id="rec"></span><span id="stale"></span>
  <label>window <select id="win">
    <option value="2">2 s</option><option value="5">5 s</option>
    <option value="10">10 s</option><option value="30" selected>30 s</option>
    <option value="60">60 s</option><option value="120">2 min</option>
  </select></label>
  <label>keep <select id="buf">
    <option value="auto" selected>auto</option>
    <option value="60">1 min</option><option value="180">3 min</option><option value="300">5 min</option>
  </select></label>
  <button id="pause">Pause</button>
  <button id="fitall">Fit all</button>
  <button id="reset">Reset layout</button>
  <span id="when"></span>
  <span id="cursor" style="color:var(--dim)"></span>
</header>
<div id="mapwrap"><canvas id="map"></canvas></div>
<div id="plots"></div>
<div id="hidden"></div>
<script>
"use strict";
// Channel registry. `alt` is the uncalibrated fallback used when the
// calibrated one is absent, so a front camera reads in degrees once lever_px
// is set and in pixels before that.
const CHANNELS=[
 {id:'wsa',   title:'Wing stroke amplitude (bottom)', unit:'deg',
  keys:[{k:'bottom/left',c:'#e55',lbl:'L'},{k:'bottom/right',c:'#5be',lbl:'R'}], def:[0,180]},
 {id:'lmr',   title:'L − R  (bottom)', unit:'deg', keys:[{k:'L-R',c:'#fd5',lbl:'L−R'}], def:[-60,60]},
 {id:'abd',   title:'Abdomen bending (bottom)', unit:'deg',
  keys:[{k:'bottom/abdomen',c:'#f88',lbl:'abd'}], def:[-45,45]},
 {id:'legs',  title:'Hind legs (bottom)', unit:'deg',
  keys:[{k:'bottom/leg_l3',c:'#6af',lbl:'L3'},{k:'bottom/leg_r3',c:'#adf',lbl:'R3'}], def:[-120,120]},
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
 {id:'edges', title:'Edges found per annulus', unit:'', hidden:true,
  keys:[{k:'bottom/abdomen_n_edges',c:'#f88',lbl:'abd'},
        {k:'bottom/leg_l3_n_edges',c:'#6af',lbl:'L3'},
        {k:'bottom/leg_r3_n_edges',c:'#adf',lbl:'R3'},
        {k:'bottom/left_n_edges',c:'#8f8',lbl:'L'},
        {k:'bottom/right_n_edges',c:'#cfc',lbl:'R'}], def:[0,4], step:true},
 {id:'lag',   title:'Camera lag', unit:'ms', hidden:true,
  keys:[{k:'bottom/lag_ms',c:'#999',lbl:'bottom'},{k:'front/lag_ms',c:'#c99',lbl:'front'}], def:[0,60]},
];
const LS='kinefly2.dash.v1';
let cfg=load();
function load(){ try{const o=JSON.parse(localStorage.getItem(LS)); if(o&&o.panels) return reconcile(o);}catch(e){} return blank(); }
function newPanel(c){ return {id:c.id,lo:c.def[0],hi:c.def[1],show:c.hidden?false:null}; }
function blank(){ return {win:30, buf:'auto', panels:CHANNELS.map(newPanel)}; }
// A saved layout predates any channel added since. Without this the new one
// would never appear and could not even be un-hidden.
function reconcile(o){
  for(const c of CHANNELS) if(!o.panels.some(p=>p.id===c.id)) o.panels.push(newPanel(c));
  if(!o.buf) o.buf='auto';
  return o;
}
function save(){ try{localStorage.setItem(LS,JSON.stringify(cfg));}catch(e){} }
function specOf(id){ return CHANNELS.find(c=>c.id===id); }

const data=[]; let marks=[], n=0, rate={t:performance.now(),n:0};
let mode='live', recName=null, recT0=null, cursorT=null;
// The view: `right` is the timestamp at the right edge of every panel, and
// `follow` means it tracks the newest sample. Everything about navigation is
// these three numbers.
const view={win:cfg.win||30, right:0, follow:true};
const root=document.getElementById('plots'), hiddenBox=document.getElementById('hidden');
const mapCv=document.getElementById('map');

function bufSec(){
  if(mode==='file') return Infinity;
  if(cfg.buf==='auto') return recName?300:60;   // a recording's worth, else just enough to glance back
  return +cfg.buf;
}
function bounds(){
  const first=data.length?data[0].t:0, last=data.length?data[data.length-1].t:0;
  // The sweep starts at the recording, when there is one and it is still in
  // the buffer; otherwise at the oldest sample we hold.
  const sweep=(recT0!=null&&recT0>=first&&recT0<=last)?recT0:first;
  return {first,last,sweep};
}
// While the data span is shorter than the window the right edge stays pinned
// one window after the start, so the trace fills the panel from the LEFT and
// grows rightwards; once it outgrows the window this becomes `last` and it
// scrolls, with no discontinuity at the changeover.
function rightEdge(){
  const b=bounds();
  if(view.follow) return Math.max(b.sweep+view.win, b.last);
  const lo=Math.min(b.first+view.win, b.last), hi=b.last;
  return Math.min(Math.max(view.right, Math.min(lo,hi)), Math.max(hi,lo));
}
function setRight(t){ view.follow=false; view.right=t; draw(); }
function setFollow(on){
  view.follow=on; if(!on) view.right=rightEdge();
  const b=document.getElementById('pause');
  b.classList.toggle('on',!on); b.textContent=on?'Pause':'Live';
  draw();
}

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

// ---- traces ----------------------------------------------------------------
// The polyline(s) one trace draws in one panel, as segments of [x, value]
// points. Consecutive samples are ALWAYS joined: zoomed in, most pixel columns
// hold no sample at all, and lifting the pen at every such column is what made
// every trace vanish on zooming in. Only a missing value breaks the line. With
// more samples than columns, a column keeps its first, lowest, highest and last
// value, so the envelope -- and a one-sample spike -- survives decimation.
function tracePath(rows,key,R,win,W){
  const segs=[]; let seg=null, col=null;
  function flush(){
    if(!col)return;
    seg.push([col.x,col.first]);
    if(col.n>1){
      if(col.iLo<col.iHi) seg.push([col.x,col.lo],[col.x,col.hi]);
      else seg.push([col.x,col.hi],[col.x,col.lo]);
      seg.push([col.x,col.last]);
    }
    col=null;
  }
  for(let i=0;i<rows.length;i++){
    const r=rows[i], dt=R-r.t; if(dt<0||dt>win)continue;
    const v=val(r,key);
    if(v==null||!isFinite(v)){ flush(); seg=null; continue; }
    const x=W*(1-dt/win), xc=Math.floor(x);
    if(!seg){ seg=[]; segs.push(seg); }
    if(col&&col.xc===xc){
      col.n++; if(v<col.lo){col.lo=v;col.iLo=col.n;} if(v>col.hi){col.hi=v;col.iHi=col.n;}
      col.last=v; continue;
    }
    flush();
    col={xc:xc,x:x,first:v,lo:v,hi:v,last:v,n:1,iLo:1,iHi:1};
  }
  flush();
  return segs;
}

// ---- time formatting -------------------------------------------------------
const TICKS=[0.1,0.2,0.5,1,2,5,10,15,30,60,120,300,600];
function tickStep(win){ for(const s of TICKS) if(win/s<=8) return s; return 900; }
function mmss(sec){
  const neg=sec<0; let s=Math.abs(sec);
  const dec=view.win<=10?1:0;
  let m=Math.floor(s/60), r=s-60*m;
  if(+r.toFixed(dec)>=60){r=0;m+=1;}
  return (neg?'-':'')+m+':'+(r<10?'0':'')+r.toFixed(dec);
}

// ---- rows in ---------------------------------------------------------------
function addRows(rows){
  for(const r of rows){
    // run_rig restarted: its frame clock begins again near zero. Kept, the old
    // run would hold the window where the new samples are not, and nothing
    // would draw -- so start afresh.
    if(mode==='live'&&data.length&&r.t<data[data.length-1].t-1){
      data.length=0; marks=[]; recT0=null;
    }
    const prev=data.length?data[data.length-1]:null;
    if(prev){
      // `stim` is HELD between changes, so a change between consecutive rows
      // is a stimulus onset.
      if(r.stim!==prev.stim&&r.stim!=null) marks.push({t:r.t,kind:'stim',label:String(r.stim)});
      if(r.rec!==prev.rec){
        if(r.rec){marks.push({t:r.t,kind:'rec',label:'REC '+r.rec}); recT0=r.t;}
        else marks.push({t:r.t,kind:'rec',label:'stop'});
      }
    } else if(r.rec!=null){ recT0=r.t; }
    if(r.rec!==recName){ recName=r.rec||null; showRec(); }
    data.push(r); n++;
  }
  const lim=bufSec();
  if(isFinite(lim)&&data.length){
    const last=data[data.length-1];
    while(data.length&&last.t_host-data[0].t_host>lim) data.shift();
    while(marks.length&&data.length&&marks[0].t<data[0].t) marks.shift();
  }
}
function showRec(){
  const el=document.getElementById('rec');
  if(recName){el.textContent='REC '+recName;el.style.display='inline';}
  else el.style.display='none';
}

// ---- panels ----------------------------------------------------------------
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
      el.onchange=()=>{const v=parseFloat(el.value); if(isFinite(v)){p[f]=v;save();draw();}};
    const nm=d.querySelector('.name');
    nm.ondragstart=e=>{e.dataTransfer.setData('text/plain',p.id);d.classList.add('drag');};
    nm.ondragend=()=>d.classList.remove('drag');
    d.ondragover=e=>e.preventDefault();
    d.ondrop=e=>{e.preventDefault(); const from=e.dataTransfer.getData('text/plain');
      if(!from||from===p.id)return; const a=cfg.panels.findIndex(x=>x.id===from), b=cfg.panels.findIndex(x=>x.id===p.id);
      const [m]=cfg.panels.splice(a,1); cfg.panels.splice(b,0,m); save(); build();};
    wireTime(d.querySelector('canvas'));
    root.appendChild(d); p.el=d;
  }
}
// Dragging a trace pans time, like pulling the paper through a chart recorder;
// moving without a button down reads the cursor out.
function wireTime(cv){
  let drag=null;
  cv.onmousedown=e=>{drag={x:e.clientX,right:rightEdge(),w:cv.getBoundingClientRect().width};
    view.follow=false; e.preventDefault();};
  window.addEventListener('mouseup',()=>{drag=null;});
  cv.onmousemove=e=>{
    const r=cv.getBoundingClientRect();
    if(drag){ setRight(drag.right-(e.clientX-drag.x)/drag.w*view.win); return; }
    cursorT=rightEdge()-view.win*(1-(e.clientX-r.left)/r.width); draw();
  };
  cv.onmouseleave=()=>{cursorT=null;draw();};
  cv.onwheel=e=>{
    e.preventDefault();
    const r=cv.getBoundingClientRect(), f=(e.clientX-r.left)/r.width;
    const R=rightEdge(), tAt=R-view.win*(1-f);
    const span=data.length?Math.max(data[data.length-1].t-data[0].t,1):60;
    view.win=Math.min(Math.max(view.win*(e.deltaY>0?1.3:1/1.3),0.5),Math.max(span,1));
    cfg.win=view.win; save();
    view.follow=false; view.right=tAt+view.win*(1-f); draw();
  };
}
function zoom(p,f){const c=(p.lo+p.hi)/2,h=(p.hi-p.lo)/2*f;p.lo=c-h;p.hi=c+h;sync(p);}
function pan(p,f){const s=(p.hi-p.lo)*f;p.lo+=s;p.hi+=s;sync(p);}
function sync(p){ if(!p.el)return; p.el.querySelector('.lo').value=p.lo.toFixed(2); p.el.querySelector('.hi').value=p.hi.toFixed(2); draw(); }
function autofit(p){
  const spec=specOf(p.id); let lo=Infinity,hi=-Infinity; const R=rightEdge();
  for(const kk of spec.keys){ const key=activeKey(kk).k;
    for(const r of data){ const dt=R-r.t; if(dt<0||dt>view.win)continue; const v=val(r,key);
      if(v!=null&&isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);} } }
  if(!isFinite(lo))return;
  const pad=(hi-lo)*0.12||Math.abs(hi)*0.1||1; p.lo=lo-pad; p.hi=hi+pad; sync(p);
}
function nice(v){ const a=Math.abs(v); return a>=100?v.toFixed(0):a>=10?v.toFixed(1):v.toFixed(2); }
const MARKC={stim:'#fa6',rec:'#f55',event:'#6cf'};

function draw(){
  const R=rightEdge(), t0=bounds().sweep, step=tickStep(view.win);
  document.getElementById('when').textContent=
    (data.length?mmss(R-t0)+(view.follow?'':'  (scrolled back)'):'');
  drawMap(R,t0);
  let labelled=false;
  for(const p of cfg.panels){
    if(p.show===false||!p.el)continue;
    const spec=specOf(p.id), cv=p.el.querySelector('canvas');
    const W=cv.width=cv.clientWidth*devicePixelRatio, H=cv.height=cv.clientHeight*devicePixelRatio;
    const g=cv.getContext('2d'); g.clearRect(0,0,W,H);
    const Y=v=>H*(1-(v-p.lo)/(p.hi-p.lo));
    const X=t=>W*(1-(R-t)/view.win);
    // grid: 4 horizontal divisions, vertical lines on real time
    g.strokeStyle='#282828'; g.lineWidth=1; g.beginPath();
    for(let i=1;i<4;i++){const y=H*i/4;g.moveTo(0,y);g.lineTo(W,y);}
    for(let t=Math.ceil((R-view.win-t0)/step)*step+t0;t<=R;t+=step){const x=X(t);g.moveTo(x,0);g.lineTo(x,H);}
    g.stroke();
    if(p.lo<0&&p.hi>0){g.strokeStyle='#444';g.beginPath();g.moveTo(0,Y(0));g.lineTo(W,Y(0));g.stroke();}
    // markers, on EVERY panel: the whole point is reading one channel against
    // the moment the stimulus changed without eyeballing across plots
    for(const m of marks){
      if(m.t<R-view.win||m.t>R)continue;
      const x=X(m.t);
      g.strokeStyle=MARKC[m.kind]||'#888'; g.globalAlpha=.55; g.lineWidth=devicePixelRatio;
      g.setLineDash(m.kind==='stim'?[4,3]:[]); g.beginPath(); g.moveTo(x,0); g.lineTo(x,H); g.stroke();
      g.setLineDash([]); g.globalAlpha=1;
      if(!labelled){ g.fillStyle=MARKC[m.kind]||'#888';
        g.font=`${10*devicePixelRatio}px ui-monospace,monospace`;
        g.fillText(m.label,x+3,10*devicePixelRatio); }
    }
    labelled=true;
    let unitShown=spec.unit;
    spec.keys.forEach(kk=>{
      const act=activeKey(kk); if(act.unit)unitShown=act.unit;
      // Min/max decimation where samples outnumber pixel columns (a 60 s window
      // at 90 Hz is 5400 points), a plain polyline where they do not.
      const segs=tracePath(data,act.k,R,view.win,W);
      g.strokeStyle=kk.c; g.lineWidth=1.4*devicePixelRatio; g.beginPath();
      for(const sg of segs){
        g.moveTo(sg[0][0],Y(sg[0][1]));
        if(sg.length===1){ g.lineTo(sg[0][0]+devicePixelRatio,Y(sg[0][1])); continue; }  // a lone sample: a dot
        for(let j=1;j<sg.length;j++){
          if(spec.step) g.lineTo(sg[j][0],Y(sg[j-1][1]));   // hold the value until it changes
          g.lineTo(sg[j][0],Y(sg[j][1]));
        }
      }
      g.stroke();
    });
    // axis labels
    g.fillStyle='#777'; g.font=`${10*devicePixelRatio}px ui-monospace,monospace`;
    g.fillText(nice(p.hi),3,11*devicePixelRatio);
    g.fillText(nice(p.lo),3,H-3);
    for(let t=Math.ceil((R-view.win-t0)/step)*step+t0;t<=R;t+=step){
      const x=X(t); if(x<18*devicePixelRatio)continue;
      g.fillText(mmss(t-t0),x+2,H-3);
    }
    if(cursorT!=null&&cursorT>=R-view.win&&cursorT<=R){ const x=X(cursorT);
      g.strokeStyle='#999';g.setLineDash([3,3]);g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();g.setLineDash([]); }
    // current values
    let at=null;
    const want=cursorT!=null?cursorT:R;
    let bd=1e9; for(const r of data){const d=Math.abs(r.t-want); if(d<bd){bd=d;at=r;}}
    p.el.querySelector('.cur').innerHTML = spec.keys.map(kk=>{
      const act=activeKey(kk); const v=at?val(at,act.k):null;
      return `<span style="color:${kk.c}">${kk.lbl} ${v==null?'–':nice(v)}</span>`;
    }).join(' &nbsp; ')+` <span style="color:var(--dim)">${unitShown}</span>`;
  }
  document.getElementById('cursor').textContent =
    cursorT!=null&&data.length?('cursor '+mmss(cursorT-t0)):'';
}

// ---- overview strip --------------------------------------------------------
// The whole buffer at a glance with the visible window as a box: this is how
// you find the stimulus you want without scrubbing blindly.
function drawMap(R,t0){
  const W=mapCv.width=mapCv.clientWidth*devicePixelRatio;
  const H=mapCv.height=mapCv.clientHeight*devicePixelRatio;
  const g=mapCv.getContext('2d'); g.clearRect(0,0,W,H);
  const b=bounds(); const span=Math.max(b.last-b.first,1e-6);
  const X=t=>W*(t-b.first)/span;
  // one representative trace: the first visible panel's first channel
  const p=cfg.panels.find(q=>q.show!==false&&q.el);
  if(p&&data.length){
    const spec=specOf(p.id), key=activeKey(spec.keys[0]).k;
    let lo=Infinity,hi=-Infinity;
    for(const r of data){const v=val(r,key); if(v!=null&&isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
    if(isFinite(lo)){
      const Y=v=>H-2-(H-4)*(v-lo)/((hi-lo)||1);
      g.strokeStyle=spec.keys[0].c; g.lineWidth=devicePixelRatio; g.globalAlpha=.85; g.beginPath();
      const cols=new Array(Math.ceil(W)).fill(null);
      for(const r of data){const v=val(r,key); if(v==null||!isFinite(v))continue;
        const x=Math.floor(X(r.t)); if(x<0||x>=cols.length)continue;
        const c=cols[x]; if(c==null)cols[x]={lo:v,hi:v}; else {c.lo=Math.min(c.lo,v);c.hi=Math.max(c.hi,v);} }
      let pen=false;
      for(let x=0;x<cols.length;x++){const c=cols[x]; if(c==null){pen=false;continue;}
        if(!pen){g.moveTo(x,Y(c.hi));pen=true;} else {g.lineTo(x,Y(c.hi));g.lineTo(x,Y(c.lo));}}
      g.stroke(); g.globalAlpha=1;
    }
  }
  for(const m of marks){ const x=X(m.t);
    g.strokeStyle=MARKC[m.kind]||'#888'; g.globalAlpha=.5; g.beginPath(); g.moveTo(x,0); g.lineTo(x,H); g.stroke(); }
  g.globalAlpha=1;
  const x0=X(R-view.win), x1=X(R);
  g.fillStyle='rgba(255,255,255,.10)'; g.fillRect(x0,0,Math.max(x1-x0,2),H);
  g.strokeStyle=view.follow?'#4c6':'#aaa'; g.lineWidth=devicePixelRatio;
  g.strokeRect(x0,0.5,Math.max(x1-x0,2),H-1);
}
(function wireMap(){
  let drag=false;
  const jump=e=>{
    const r=mapCv.getBoundingClientRect(), b=bounds();
    const t=b.first+(b.last-b.first)*(e.clientX-r.left)/r.width;
    // Dropping the box at the far right re-arms live follow.
    if(t>=b.last-view.win*0.05){ setFollow(true); return; }
    setRight(t+view.win/2);
  };
  mapCv.onmousedown=e=>{drag=true;jump(e);e.preventDefault();};
  mapCv.onmousemove=e=>{if(drag)jump(e);};
  window.addEventListener('mouseup',()=>{drag=false;});
})();

// ---- controls --------------------------------------------------------------
document.getElementById('win').onchange=e=>{view.win=cfg.win=+e.target.value;save();draw();};
document.getElementById('buf').onchange=e=>{cfg.buf=e.target.value;save();};
document.getElementById('pause').onclick=()=>setFollow(!view.follow);
document.getElementById('fitall').onclick=()=>{cfg.panels.forEach(p=>{if(p.show!==false)autofit(p);});save();draw();};
document.getElementById('reset').onclick=()=>{cfg=blank();view.win=cfg.win;save();build();draw();};
document.getElementById('win').value=String(cfg.win);
document.getElementById('buf').value=cfg.buf;
window.addEventListener('keydown',e=>{
  if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;
  const b=bounds();
  if(e.code==='Space'){setFollow(!view.follow);e.preventDefault();}
  else if(e.key==='ArrowLeft'){setRight(rightEdge()-view.win/2);e.preventDefault();}
  else if(e.key==='ArrowRight'){setRight(rightEdge()+view.win/2);e.preventDefault();}
  else if(e.key==='Home'){setRight(b.first+view.win);}
  else if(e.key==='End'){setFollow(true);}
});

// ---- data in ---------------------------------------------------------------
function autoHide(){
  if(window._autoHid||data.length<60)return; window._autoHid=true;
  let ch=false; for(const p of cfg.panels){ if(p.show===null){ p.show=!!hasData(specOf(p.id)); ch=true; } }
  if(ch){save();build();}
}
fetch('/meta').then(r=>r.json()).then(meta=>{
  mode=meta.mode;
  if(mode==='file'){
    document.getElementById('title').textContent=meta.name||'recording';
    document.getElementById('status').textContent=
      `${meta.n} rows, ${mmss(meta.span)} · ${meta.code_version||'unknown code'}`
      +(meta.primary_known===false?`  ·  primary camera guessed (${meta.primary})`:'');
    for(const ev of meta.events||[]) marks.push({t:ev.t,kind:'event',label:ev.text.slice(0,40)});
    fetch('/history').then(r=>r.json()).then(rows=>{
      addRows(rows); marks.sort((a,b)=>a.t-b.t);
      autoHide(); setRight(bounds().first+view.win);   // start at the beginning of the file
    });
  } else {
    startStream();
  }
});
function startStream(){
  const es=new EventSource('/stream');
  es.onmessage=e=>{
    let rows; try{rows=JSON.parse(e.data);}catch(err){document.getElementById('status').textContent='bad frame';return;}
    addRows(rows);
    const dt=(performance.now()-rate.t)/1000;
    if(dt>2){document.getElementById('status').textContent=`${((n-rate.n)/dt).toFixed(0)} states/s`;rate={t:performance.now(),n:n};}
    // A camera too old to trust: its traces are already blank, and this says the
    // gap is a stalled camera rather than a fly holding still.
    const sb=document.getElementById('stale'), last=data[data.length-1];
    if(last&&last.stale){ sb.textContent=`${last.stale} STALE`; sb.style.display='inline'; }
    else sb.style.display='none';
    // First data in: hide panels that carry nothing, so a one-camera session
    // is not half empty. The user's own choices are never overridden.
    autoHide();
  };
  es.onerror=()=>{document.getElementById('status').textContent='disconnected';};
}
build(); setInterval(()=>{if(view.follow)draw();},50);
window.addEventListener('resize',()=>draw());
</script></body></html>
"""


def make_handler(src):
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
            elif self.path == '/meta':
                self._send(json.dumps(src.meta()).encode(), 'application/json')
            elif self.path == '/history':
                self._send(json.dumps(src.all()).encode(), 'application/json')
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                t = -math.inf
                try:
                    rows = src.all()
                    if rows:
                        self.wfile.write(('data: %s\n\n' % json.dumps(rows)).encode())
                        t = rows[-1]['t_host']
                    while True:
                        time.sleep(0.05)
                        rows = src.since(t)
                        if rows:
                            t = rows[-1]['t_host']
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
    ap.add_argument('--file', metavar='H5',
                    help='View a saved recording (<run>_flystate.h5) instead of the live stream')
    ap.add_argument('--history', type=float, default=MAX_HISTORY_S, metavar='S',
                    help='Seconds of live history to keep for scrolling back (default %d)'
                         % MAX_HISTORY_S)
    args = ap.parse_args()

    if args.file:
        src = FileBuffer(args.file)
        info = src.info
        print('Dashboard: http://localhost:%d' % args.port)
        print('  file:   %s' % args.file)
        print('  %d rows, %.1f s, primary %s, recorded by %s'
              % (info['n'], info['span'], info['primary'] or '?',
                 info.get('code_version') or 'unknown code'))
        if not info['primary_known']:
            print('  note: this file predates the `rig` attribute, so the primary camera\n'
                  '        was GUESSED as %r. It decides whose frame clock the time axis\n'
                  '        is; if that is the wrong camera the axis is wrong (section 2.4).'
                  % info['primary'])
    else:
        src = StateBuffer(args.zmq, history_s=args.history)
        print('Dashboard: http://localhost:%d  (fly state from %s)' % (args.port, args.zmq))
    server = ThreadingHTTPServer(('', args.port), make_handler(src))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
