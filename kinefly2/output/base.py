"""Output sink interface.

Sinks receive every MergedFlyState synchronously in the primary camera's
processing thread. handle() must be fast (sub-millisecond): either do trivial
work inline (a short serial write) or enqueue to an internal thread and return.

The future Reiser LED-panel closed-loop driver attaches here as another sink;
nothing in the core needs to change.
"""


class FlystateSink:
    def handle(self, merged):
        raise NotImplementedError

    def close(self):
        pass
