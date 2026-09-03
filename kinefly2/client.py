"""Python client for experiment scripts.

    from kinefly2.client import Kinefly2Client
    k = Kinefly2Client('localhost')                  # or the rig's IP
    k.status()
    with k.recording(fly_id='017', protocol='YAMV', notes='stripe fixation'):
        k.event({'trial': 1, 'event': 'stim_on'})
        ...                                          # run the trial
        k.event({'trial': 1, 'event': 'stim_off'})
    # recording is stopped on leaving the block (also on exceptions)

    for state in k.states():                         # live fly state, ~camera rate
        parts = state['cameras']['bottom']['parts']
        lmr = parts['left']['angles'][0] - parts['right']['angles'][0]

Only pyzmq is needed on the client side (pip install pyzmq); the client
does not import the rest of kinefly2, so this file can be copied anywhere.
"""
import contextlib
import json

import zmq


class Kinefly2Error(RuntimeError):
    pass


class Kinefly2Client:
    def __init__(self, host='localhost', command_port=5558, state_port=5556,
                 timeout_s=5.0):
        self.host = host
        self.command_port = int(command_port)
        self.state_port = int(state_port)
        self.timeout_ms = int(timeout_s * 1000)
        self.context = zmq.Context.instance()
        self._sock = None

    # ------------------------------------------------------------ transport

    def _socket(self):
        if self._sock is None:
            s = self.context.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            s.connect('tcp://%s:%d' % (self.host, self.command_port))
            self._sock = s
        return self._sock

    def request(self, **req):
        s = self._socket()
        try:
            s.send_string(json.dumps(req))
            reply = json.loads(s.recv_string())
        except zmq.Again:
            # A REQ socket is unusable after a timeout; rebuild it.
            s.close(linger=0)
            self._sock = None
            raise Kinefly2Error('no reply from Kinefly2 at %s:%d (is run_rig running?)'
                                % (self.host, self.command_port))
        if not reply.get('ok', False):
            raise Kinefly2Error(reply.get('error', 'unknown error'))
        return reply

    # ------------------------------------------------------------- commands

    def ping(self):
        return self.request(cmd='ping')

    def status(self):
        return self.request(cmd='status')

    def start(self, fly_id=None, protocol=None, genotype=None, notes=None,
              record_video=None, **fields):
        """Start a recording; returns {'name', 'output_dir'}."""
        return self.request(cmd='start', fly_id=fly_id, protocol=protocol,
                            genotype=genotype, notes=notes, record_video=record_video,
                            **fields)

    def stop(self):
        """Stop the recording; returns its summary."""
        return self.request(cmd='stop')['summary']

    def event(self, text):
        """Log an event (dict/list are JSON-encoded) on the recording's clock."""
        if not isinstance(text, str):
            text = json.dumps(text)
        return self.request(cmd='event', text=text)

    def snapshot(self):
        """Save a PNG of each camera's current frame; returns the paths."""
        return self.request(cmd='snapshot')['paths']

    @contextlib.contextmanager
    def recording(self, **fields):
        """with k.recording(fly_id='017', protocol='YAMV'): ...  (stops on exit)."""
        info = self.start(**fields)
        try:
            yield info
        finally:
            try:
                self.stop()
            except Kinefly2Error:
                pass

    # ---------------------------------------------------------- live state

    def states(self, timeout_s=None):
        """Generator over the live merged fly state (JSON dicts)."""
        sub = self.context.socket(zmq.SUB)
        sub.connect('tcp://%s:%d' % (self.host, self.state_port))
        sub.setsockopt_string(zmq.SUBSCRIBE, '')
        if timeout_s is not None:
            sub.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        try:
            while True:
                try:
                    yield json.loads(sub.recv_string())
                except zmq.Again:
                    return
        finally:
            sub.close(linger=0)

    def close(self):
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
