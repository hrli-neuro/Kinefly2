"""ZMQ REP command socket: JSON request in, JSON reply out.

Experiment scripts use kinefly2.client.Kinefly2Client (or apps/ctl.py) to
start/stop recordings, log events, query status and take snapshots while
Kinefly2 keeps tracking. Requests are dispatched to a handler callable
(RecordingSession.handle_command) from a server thread.
"""
import json
import threading

import zmq


class CommandServer:
    def __init__(self, handler, port=5558, bind='tcp://*'):
        self.handler = handler
        self.port = int(port)
        self.n_requests = 0
        self._stop = False
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('%s:%d' % (bind, self.port))
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='command-server')
        self.thread.start()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop:
            if not poller.poll(200):
                continue
            raw = self.socket.recv_string()
            self.n_requests += 1
            try:
                req = json.loads(raw)
                if not isinstance(req, dict):
                    raise ValueError('request must be a JSON object')
                reply = self.handler(req)
            except Exception as e:
                reply = {'ok': False, 'error': str(e)}
            self.socket.send_string(json.dumps(reply, default=str))

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.socket.close(linger=0)
