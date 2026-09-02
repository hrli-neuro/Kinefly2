"""Receive stimulus/experiment events from other programs and log them with
the fly state, on a shared clock.

Any program (the arena/stimulus software, a trial controller) sends a string
(JSON recommended) over a ZMQ PUSH socket:

    import zmq
    sock = zmq.Context().socket(zmq.PUSH)
    sock.connect('tcp://localhost:5557')          # rig IP from another machine
    sock.send_string('{"trial": 12, "event": "closed_loop_start", "gain": 1.5}')

Each event is stamped on arrival with t_host (monotonic, same clock as the
fly-state rows) and t_wall (time.time(), for aligning with the sender's own
timestamps), written to the HDF5 /events group and to <name>_events.jsonl.
"""
import json
import threading
import time

import zmq


class EventReceiver:
    def __init__(self, port, logger=None, jsonl_path=None, bind='tcp://*'):
        self.port = int(port)
        self.logger = logger
        self.jsonl = open(jsonl_path, 'a') if jsonl_path else None
        self.n_events = 0
        self._stop = False
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind('%s:%d' % (bind, self.port))
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='event-receiver')
        self.thread.start()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop:
            if not poller.poll(200):
                continue
            text = self.socket.recv_string()
            t_host = time.monotonic()
            t_wall = time.time()
            self.n_events += 1
            if self.logger is not None:
                self.logger.log_event(text, t_host=t_host, t_wall=t_wall)
            if self.jsonl is not None:
                self.jsonl.write(json.dumps({'t_host': t_host, 't_wall': t_wall,
                                             'text': text}) + '\n')
                self.jsonl.flush()

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.socket.close(linger=0)
        if self.jsonl is not None:
            self.jsonl.close()
