"""Optional, bounded runtime observations; never evidence for motion permission."""

import math
import threading
import time


MUX_METRICS = frozenset({
    'control_receipt_age', 'control_callback', 'cmd_vel_publish',
    'status_publish', 'ownership_query', 'output_gap', 'status_gap', 'reset_tick_gap',
})
GUARD_METRICS = frozenset({'control_publish', 'tick_gap', 'ownership_query', 'tf_lookup'})


class RuntimeTiming:
    """Last/max/count only, with fixed keys and one slow warning per second.

    Durations use the local steady clock. A packet's ROS age is kept separate;
    it includes source publish, DDS delivery and executor waiting, not just wire
    time. None of these observations refreshes a command or sensor timestamp.
    """

    def __init__(self, names, *, clock=time.monotonic, warning=None):
        self.names = frozenset(names)
        self.clock = clock
        self.warning = warning
        self.metrics = {}
        self.previous = {}
        self.last_warning_at = -math.inf
        self._lock = threading.Lock()

    def observe(self, name, value, *, warning_excess=None):
        if name not in self.names or type(value) not in (int, float) or not math.isfinite(value):
            return
        value = float(value)
        warn = False
        with self._lock:
            metric = self.metrics.get(name)
            if metric is None:
                self.metrics[name] = {'count': 1, 'last_seconds': value, 'max_seconds': value}
            else:
                metric['count'] += 1
                metric['last_seconds'] = value
                metric['max_seconds'] = max(metric['max_seconds'], value)
            slow = abs(value) if warning_excess is None else warning_excess
            now = self.clock()
            if self.warning is not None and slow >= .1 and now-self.last_warning_at >= 1.:
                self.last_warning_at = now
                warn = True
        if warn:
            try:
                self.warning(f'Runtime timing: {name}={value:.3f}s (diagnostic only)')
            except Exception:
                pass  # Observability must not interrupt arbitration or a stop.

    def gap(self, name, expected_period):
        now = self.clock()
        with self._lock:
            previous = self.previous.get(name)
            if name in self.names:
                self.previous[name] = now
        if previous is not None:
            elapsed = max(0., now-previous)
            self.observe(name, elapsed, warning_excess=max(0., elapsed-expected_period))

    def snapshot(self):
        with self._lock:
            return {name: dict(metric) for name, metric in self.metrics.items()}


def timed_call(timing, name, callback, *args, **kwargs):
    if timing is None:
        return callback(*args, **kwargs)
    started = timing.clock()
    try:
        return callback(*args, **kwargs)
    finally:
        timing.observe(name, max(0., timing.clock()-started))


def guard_timing_report(payload):
    """Copy only bounded numerical guard observations from an incoming packet."""
    if not isinstance(payload, dict):
        return None
    report = payload.get('timing_diagnostics')
    if not isinstance(report, dict):
        return None
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        return None
    clean = {}
    for name in GUARD_METRICS:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            continue
        count, last, maximum = (metric.get(key) for key in ('count', 'last_seconds', 'max_seconds'))
        if (type(count) is int and 0 < count < 2**63
                and type(last) in (float, int) and type(maximum) in (float, int)
                and math.isfinite(last) and math.isfinite(maximum)
                and 0 <= last <= maximum <= 86400.):
            clean[name] = {'count': count, 'last_seconds': last, 'max_seconds': maximum}
    return clean or None
