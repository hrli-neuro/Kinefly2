"""Receive stimulus/experiment events from other programs and log them with
the fly state, on a shared clock.

Any program (the arena/stimulus software, a trial controller) sends a string
(JSON recommended) over a ZMQ PUSH socket (fire-and-forget), or via the
command socket's 'event' command (acknowledged):

    import zmq
    sock = zmq.Context().socket(zmq.PUSH)
    sock.connect('tcp://localhost:5557')          # rig IP from another machine
    sock.send_string('{"trial": 12, "event": "closed_loop_start", "gain": 1.5}')

The receiver is bound once for the lifetime of run_rig; a recording session
attaches its logger + jsonl file and detaches them when it stops. Events that
arrive while no recording is active are counted and dropped. Each logged
event is stamped with t_host (monotonic, same clock as the fly-state rows)
and t_wall (time.time(), for aligning with the sender's own timestamps).
"""
import json
import threading
import time

import zmq


class EventReceiver:
    def __init__(self, port, logger=None, jsonl_path=None, bind='tcp://*'):
        self.port = int(port)
        self.logger = None
        self.jsonl = None
        self.n_events = 0
        self.n_dropped_idle = 0
        self._lock = threading.Lock()
        self._stop = False
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind('%s:%d' % (bind, self.port))
        if logger is not None or jsonl_path:
            self.attach(logger, jsonl_path)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='event-receiver')
        self.thread.start()

    def attach(self, logger, jsonl_path=None):
        with self._lock:
            self.logger = logger
            self.jsonl = open(jsonl_path, 'a') if jsonl_path else None

    def detach(self):
        with self._lock:
            self.logger = None
            if self.jsonl is not None:
                self.jsonl.close()
                self.jsonl = None

    def log(self, text, t_host=None, t_wall=None):
        """Log an event now. Returns True if a recording captured it."""
        t_host = time.monotonic() if t_host is None else t_host
        t_wall = time.time() if t_wall is None else t_wall
        with self._lock:
            if self.logger is None and self.jsonl is None:
                self.n_dropped_idle += 1
                return False
            self.n_events += 1
            if self.logger is not None:
                self.logger.log_event(text, t_host=t_host, t_wall=t_wall)
            if self.jsonl is not None:
                self.jsonl.write(json.dumps({'t_host': t_host, 't_wall': t_wall,
                                             'text': text}) + '\n')
                self.jsonl.flush()
            return True

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop:
            if not poller.poll(200):
                continue
            text = self.socket.recv_string()
            self.log(text)

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.socket.close(linger=0)
        self.detach()
