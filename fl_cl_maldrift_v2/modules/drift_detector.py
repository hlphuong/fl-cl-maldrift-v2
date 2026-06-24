"""
modules/drift_detector.py
HDDM-W, ADWIN, DDM, EDDM với unified drift score [0,1].
"""
import math
import numpy as np
from collections import deque


class BaseDriftDetector:
    STATE_STABLE  = 0
    STATE_WARNING = 1
    STATE_DRIFT   = 2

    def __init__(self, window: int = 30):
        self._window = deque(maxlen=window)
        self._state  = self.STATE_STABLE
        self._raw    = 0.0

    def update(self, error: float) -> int:
        raise NotImplementedError

    def score(self) -> float:
        """Normalized drift score ∈ [0,1]."""
        if self._state == self.STATE_DRIFT:   return 1.0
        if self._state == self.STATE_WARNING: return 0.5
        self._window.append(self._raw)
        if len(self._window) < 3: return 0.0
        mu = np.mean(self._window)
        sg = np.std(self._window) + 1e-8
        return float(np.clip((self._raw - mu) / sg / 3, 0, 1))

    def reset(self):
        self._state = self.STATE_STABLE
        self._raw   = 0.0


class HDDMWDetector(BaseDriftDetector):
    """Hoeffding Drift Detection (Weighted) — exponentially weighted mean."""

    def __init__(self, drift_conf=0.001, warn_conf=0.005, window=30):
        super().__init__(window)
        self.dc   = drift_conf
        self.wc   = warn_conf
        self._lam = 0.05
        self._n   = 0
        self._mu_ref  = None
        self._mu_curr = 0.0

    def update(self, error: float) -> int:
        self._n += 1
        if self._mu_ref is None:
            self._mu_ref = error
        self._mu_curr = (1 - self._lam) * self._mu_curr + self._lam * error
        dev = abs(self._mu_curr - self._mu_ref)
        self._raw = dev
        ed = self._eps(self.dc)
        ew = self._eps(self.wc)
        if dev > ed:
            self._state  = self.STATE_DRIFT
            self._mu_ref = self._mu_curr
            self._n      = 1
        elif dev > ew:
            self._state = self.STATE_WARNING
        else:
            self._state = self.STATE_STABLE
        return self._state

    def _eps(self, conf):
        if self._n < 2: return float("inf")
        return math.sqrt(math.log(2 / conf) / (2 * self._n))


class ADWINDetector(BaseDriftDetector):
    def __init__(self, delta=0.002, window=30):
        super().__init__(window)
        self._delta = delta
        self._buf   = deque()
        self._total = 0.0

    def update(self, error: float) -> int:
        self._buf.append(error)
        self._total += error
        n = len(self._buf)
        if n < 4:
            return self.STATE_STABLE
        best = 0.0
        rs   = 0.0
        cut  = None
        for i, v in enumerate(self._buf):
            rs += v
            n0, n1 = i + 1, n - i - 1
            if n1 == 0: continue
            m0, m1 = rs / n0, (self._total - rs) / n1
            diff = abs(m0 - m1)
            eps  = math.sqrt((1 / n0 + 1 / n1) * math.log(4 * n / self._delta) / 2)
            if diff > eps and diff > best:
                best, cut = diff, i
        self._raw = best
        if cut is not None:
            rem = sum(list(self._buf)[:cut + 1])
            for _ in range(cut + 1): self._buf.popleft()
            self._total -= rem
            self._state  = self.STATE_DRIFT
        else:
            self._state = self.STATE_STABLE
        return self._state


class DDMDetector(BaseDriftDetector):
    def __init__(self, warn_lv=2.0, drift_lv=3.0, window=30):
        super().__init__(window)
        self.wl = warn_lv; self.dl = drift_lv
        self._n = 0; self._p = 1.0; self._s = 0.0
        self._pmin = float("inf"); self._ps_min = float("inf")

    def update(self, error: float) -> int:
        self._n += 1
        self._p += (error - self._p) / self._n
        self._s  = math.sqrt(max(0, self._p * (1 - self._p) / self._n))
        ps = self._p + self._s
        if ps < self._ps_min:
            self._pmin  = self._p
            self._ps_min = ps
        self._raw = ps - self._ps_min
        if self._n < 30: return self.STATE_STABLE
        if self._p + self._s > self._pmin + self.dl * self._s:
            self._state = self.STATE_DRIFT
            self._n = 0; self._p = 1.0; self._s = 0.0
            self._pmin = float("inf"); self._ps_min = float("inf")
        elif self._p + self._s > self._pmin + self.wl * self._s:
            self._state = self.STATE_WARNING
        else:
            self._state = self.STATE_STABLE
        return self._state


class EDDMDetector(BaseDriftDetector):
    def __init__(self, alpha=0.95, beta=0.9, window=30):
        super().__init__(window)
        self.alpha = alpha; self.beta = beta
        self._nerr = 0; self._last = -1
        self._dmean = 0.0; self._dvar = 0.0
        self._dmax  = 0.0; self._i    = 0

    def update(self, error: float) -> int:
        self._i += 1
        if error == 1:
            self._nerr += 1
            if self._last >= 0:
                d = self._i - self._last
                self._dmean += (d - self._dmean) / self._nerr
                self._dvar  += (d - self._dmean) ** 2
            self._last = self._i
        if self._nerr < 2: return self.STATE_STABLE
        sd     = math.sqrt(self._dvar / max(1, self._nerr - 1))
        metric = self._dmean + 2 * sd
        self._raw = 1 - metric / max(1e-8, self._dmax)
        if metric > self._dmax: self._dmax = metric
        ratio = metric / self._dmax if self._dmax else 1.0
        if ratio < self.beta:         self._state = self.STATE_DRIFT
        elif ratio < self.alpha:      self._state = self.STATE_WARNING
        else:                         self._state = self.STATE_STABLE
        return self._state


def get_detector(name: str, cfg: dict) -> BaseDriftDetector:
    name = name.lower()
    if name == "hddm_w":
        return HDDMWDetector(cfg.get("drift_confidence", 0.001),
                             cfg.get("warning_confidence", 0.005))
    elif name == "adwin":  return ADWINDetector()
    elif name == "ddm":    return DDMDetector()
    elif name == "eddm":   return EDDMDetector()
    raise ValueError(f"Unknown detector: {name}")
