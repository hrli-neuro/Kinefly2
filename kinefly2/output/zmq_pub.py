"""Publish the merged fly state on a ZMQ PUB socket as JSON.

Any client (Python, MATLAB, another machine) can subscribe:

    import zmq, json
    sock = zmq.Context().socket(zmq.SUB)
    sock.connect('tcp://localhost:5556')
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    state = json.loads(sock.recv_string())
"""
import json

import zmq

from .base import FlystateSink


class ZmqPublisher(FlystateSink):
    def __init__(self, port=5556, bind='tcp://*'):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        # Never block or buffer unboundedly: drop messages if no one is reading.
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.bind('%s:%d' % (bind, port))

    def handle(self, merged):
        try:
            self.socket.send_string(json.dumps(merged.to_dict()), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        self.socket.close(linger=0)
