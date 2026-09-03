"""Live browser dashboard for the fly state (idea 9).

Subscribes to the ZMQ state stream published by run_rig and serves one
self-contained HTML page (no external libraries, works on an offline rig
network) with live traces of L, R, L-R, head angles and wingbeat frequency.
Runs as a separate process, so it adds zero load to the tracker, and can be
opened from any machine on the network.

    python -m kinefly2.apps.dashboard                         # tracker on this machine
    python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
    then open  http://<this-machine>:8000
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

HISTORY_S = 30.0


class StateBuffer:
    def __init__(self, zmq_addr):
        self.samples = collections.deque()
        self.lock = threading.Lock()
        self.n = 0
        self.t_last = None
        self.zmq_addr = zmq_addr
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _angle(state, cam, part, i=0):
        try:
            a = state['cameras'][cam]['parts'][part]['angles']
            return a[i] if len(a) > i else None
        except KeyError:
            return None

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
            cams = list(state.get('cameras', {}))
            row = {'t': state['t_host'], 'seq': state['seq']}
            for cam in cams:
                for part in ('left', 'right', 'head', 'abdomen'):
                    row['%s/%s' % (cam, part)] = self._angle(state, cam, part)
                row['%s/head_yaw' % cam] = self._angle(state, cam, 'head', 1)
                try:
                    row['%s/wbf' % cam] = state['cameras'][cam]['parts']['aux']['freq']
                except KeyError:
                    pass
            with self.lock:
                self.samples.append(row)
                self.n += 1
                self.t_last = time.time()
                while self.samples and row['t'] - self.samples[0]['t'] > HISTORY_S:
                    self.samples.popleft()

    def since(self, t):
        with self.lock:
            return [s for s in self.samples if s['t'] > t]

    def all(self):
        with self.lock:
            return list(self.samples)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Kinefly2 live</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
h1{font-size:15px;margin:8px 12px;font-weight:600}
#status{color:#8c8;margin-left:12px}
.plot{margin:4px 12px}
canvas{width:100%;height:120px;display:block;background:#181818;border:1px solid #333}
.lbl{display:flex;justify-content:space-between;color:#aaa;font-size:12px}
</style></head><body>
<h1>Kinefly2 live <span id="status">connecting…</span></h1>
<div id="plots"></div>
<script>
const H=30; // seconds of history
const specs=[
 {title:'Wing stroke amplitude (bottom)', keys:['bottom/left','bottom/right'], colors:['#e55','#5be'], unit:'rad'},
 {title:'L − R (bottom)', keys:['L-R'], colors:['#fd5'], unit:'rad', zero:true},
 {title:'Head yaw (bottom)', keys:['bottom/head'], colors:['#8f8'], unit:'rad', zero:true},
 {title:'Head roll (front)', keys:['front/head'], colors:['#c8f'], unit:'rad', zero:true},
 {title:'Head yaw: bottom (green) vs front landmarks (purple)', keys:['bottom/head','front/head_yaw'], colors:['#8f8','#c8f'], unit:'rad', zero:true},
 {title:'Wingbeat frequency (bottom aux)', keys:['bottom/wbf'], colors:['#fa6'], unit:'Hz'},
];
const data=[]; let last=null, n=0, t0=null;
const root=document.getElementById('plots');
for(const s of specs){const d=document.createElement('div');d.className='plot';
 d.innerHTML=`<div class="lbl"><span>${s.title}</span><span class="val"></span></div><canvas></canvas>`;root.appendChild(d);s.el=d;}
function val(row,k){ if(k==='L-R'){const l=row['bottom/left'],r=row['bottom/right'];return (l==null||r==null)?null:l-r;} return row[k]==null?null:row[k]; }
function draw(){
 const now=last?last.t:0;
 for(const s of specs){
  const c=s.el.querySelector('canvas'); const W=c.width=c.clientWidth*devicePixelRatio, Hh=c.height=c.clientHeight*devicePixelRatio;
  const g=c.getContext('2d'); g.clearRect(0,0,W,Hh);
  let lo=Infinity,hi=-Infinity; const series=s.keys.map(k=>data.map(r=>[r.t,val(r,k)]));
  for(const ser of series) for(const [t,v] of ser) if(v!=null&&isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  if(!isFinite(lo)){continue;} if(s.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);} const pad=(hi-lo)*0.1||0.1; lo-=pad;hi+=pad;
  const X=t=>W*(1-(now-t)/H), Y=v=>Hh*(1-(v-lo)/(hi-lo));
  g.strokeStyle='#333';g.beginPath(); if(s.zero){g.moveTo(0,Y(0));g.lineTo(W,Y(0));} g.stroke();
  series.forEach((ser,i)=>{g.strokeStyle=s.colors[i];g.lineWidth=1.5*devicePixelRatio;g.beginPath();let pen=false;
   for(const [t,v] of ser){ if(v==null||!isFinite(v)){pen=false;continue;} const x=X(t),y=Y(v); if(!pen){g.moveTo(x,y);pen=true;} else g.lineTo(x,y);} g.stroke();});
  g.fillStyle='#888';g.font=`${11*devicePixelRatio}px system-ui`;g.fillText(hi.toFixed(2),4,12*devicePixelRatio);g.fillText(lo.toFixed(2),4,Hh-4);
  const cur=s.keys.map((k,i)=>{const v=last?val(last,k):null;return `<span style="color:${s.colors[i]}">${k}: ${v==null?'–':v.toFixed(3)} ${s.unit}</span>`;}).join(' &nbsp; ');
  s.el.querySelector('.val').innerHTML=cur;
 }
}
const es=new EventSource('/stream');
es.onmessage=e=>{const rows=JSON.parse(e.data); for(const r of rows){data.push(r);last=r;n++;}
 while(data.length&&last.t-data[0].t>H)data.shift();
 if(!t0)t0={t:performance.now(),n:0}; const dt=(performance.now()-t0.t)/1000; if(dt>2){document.getElementById('status').textContent=`${((n-t0.n)/dt).toFixed(0)} states/s`; t0={t:performance.now(),n:n};}
};
es.onerror=()=>{document.getElementById('status').textContent='disconnected';};
setInterval(draw,50);
</script></body></html>"""


def make_handler(buf):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == '/':
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/history':
                body = json.dumps(buf.all()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
