"""Reject delayed, future-dated, or reordered physical sensor samples."""


class SensorStampGuard:
    def __init__(self, max_age_sec=1.0, future_tolerance_sec=0.25):
        if max_age_sec <= 0.0 or future_tolerance_sec < 0.0:
            raise ValueError('Invalid sensor timestamp limits')
        self.max_age_ns = int(max_age_sec * 1e9)
        self.future_tolerance_ns = int(future_tolerance_sec * 1e9)
        self.last_stamp_ns = None

    def accept(self, stamp_ns, clock_ns):
        if stamp_ns <= 0 or clock_ns <= 0:
            return False
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            return False
        age_ns = clock_ns - stamp_ns
        if age_ns > self.max_age_ns or age_ns < -self.future_tolerance_ns:
            return False
        self.last_stamp_ns = stamp_ns
        return True
