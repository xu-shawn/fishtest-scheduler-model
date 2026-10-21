"""itp policies.  Each returns the internal throughput of every active run; the
scheduler then hands out cores roughly in proportion to it.  Every rule is computed
from test properties and per-user totals in the same way for everyone."""

import numpy as np


def tc_weight(s, power=0.774, cap=None):
    tcr = s.tcr if cap is None else np.minimum(s.tcr, cap)
    return np.where(tcr > 1, tcr ** power, 1.0)


def latency_bonus(s):
    """Current master: +100% at 200k games, +10%/+20% when LLR > 2."""
    return np.minimum(s.games / 200_000, 1.0) + np.where(
        s.sprt & s.llr_hi(), 0.1 + 0.1 * (s.elo0 > 0), 0.0)


def running(s):
    return (s.workers > 0).astype(float)


def active_count(s):
    """Per-user number of runs that currently have workers (as in update_itp)."""
    return s.user_sum(running(s))


def count_malus(c):
    return 36 / (36 + c ** 2)


class Current:
    """master (643fcfc8): tp * tc_ratio^0.774 * 36/(36+count^2) * (1 + bonus)."""
    name = "current"

    def itp(self, s):
        return s.tp * tc_weight(s) * count_malus(active_count(s)) * (1 + latency_bonus(s))


class PreJune:
    """Before 1a0fbfe7: LLR bonus (llr+3)/3 for llr in [0,2], x n/200k past 200k games.
    LLR paths are not in the data: passed runs ramp linearly to +2.94, others stay <= 0."""
    name = "pre_june"

    def itp(self, s):
        frac = np.where(np.isnan(s.total), 0.0, s.games / np.maximum(s.total, 1))
        llr = np.where(s.passed, 2.94 * frac, 0.0)
        b = np.where(s.sprt, (np.clip(llr, 0, 2) + 3) / 3, 1.0)
        b *= np.where(s.sprt & (s.games > 200_000), np.minimum(s.games / 200_000, 2), 1.0)
        return s.tp * tc_weight(s) * count_malus(active_count(s)) * b


class CostCount:
    """c1f29c with each running test counted by its cost weight instead of 1."""

    def __init__(self, power=0.774):
        self.power = power
        self.name = "cost_count" if power == 0.774 else f"cost_count_p{power}"

    def itp(self, s):
        c = s.user_sum(running(s) * tc_weight(s, self.power))
        return s.tp * tc_weight(s) * count_malus(c) * (1 + latency_bonus(s))


class UserCap:
    """A user's total weight (sum of tp/100 * tc_ratio^0.774 over running tests) is
    capped at S fresh STC tests; above the cap all their tests are scaled down
    together.  Replaces the count malus.  S=1 is an equal share per user."""

    def __init__(self, S, additive=False):
        self.S = S
        self.additive = additive
        self.name = f"user_cap{S:g}" + ("+add" if additive else "")

    def itp(self, s):
        base = s.tp / 100 * tc_weight(s)
        L = s.user_sum(running(s) * base)
        scale = np.minimum(1.0, self.S / np.maximum(L, 1e-9))
        b = latency_bonus(s)
        if self.additive:
            return 100 * scale * (base + b)
        return 100 * scale * base * (1 + b)


class TcCap:
    """Weight long time controls like LTC at most."""

    def __init__(self, cap=6.0):
        self.cap = cap
        self.name = f"tc_cap{cap:g}"

    def itp(self, s):
        return s.tp * tc_weight(s, cap=self.cap) * count_malus(active_count(s)) * (1 + latency_bonus(s))


class AdditiveBonus:
    """The games-based latency bonus is worth the same absolute itp for every test
    (100 * bonus * malus), instead of multiplying the TC weight."""
    name = "additive_bonus"

    def itp(self, s):
        m = count_malus(active_count(s))
        return m * (s.tp * tc_weight(s) + 100 * latency_bonus(s))


class SpecMalus:
    """Tests above STC cost with no cheaper passed test on the same branch get x f."""

    def __init__(self, f=0.5):
        self.f = f
        self.name = f"spec_malus{f:g}"

    def itp(self, s):
        return Current().itp(s) * np.where(s.spec, self.f, 1.0)


class CoreCap:
    """Current itp, plus a ceiling on cores: either for all heavy tests together
    (tc_ratio > 6.5: SMP-LTC, VLTC, VVLTC) or for any single test."""

    def __init__(self, frac, scope):
        self.frac = frac
        self.scope = scope
        self.name = f"{scope}_cap{int(round(frac * 100))}"

    def itp(self, s):
        return Current().itp(s)

    def allowed(self, sim, a, w):
        total = sim.wcores.sum()
        cw = sim.wcores[w]
        if self.scope == "heavy":
            heavy = sim.runs["tcr"][a] > 6.5
            used = sim.cores_on[a][heavy].sum()
            if used + cw > self.frac * total:
                return ~heavy
            return None
        return sim.cores_on[a] + cw <= max(self.frac * total, cw)


class CostCountAdd:
    """cost_count plus the additive latency bonus."""

    def __init__(self, power=0.5):
        self.power = power
        self.name = f"cost_count_p{power}+add"

    def itp(self, s):
        c = s.user_sum(running(s) * tc_weight(s, self.power))
        return count_malus(c) * (s.tp * tc_weight(s) + 100 * latency_bonus(s))


ALL = [Current(), PreJune(), CostCount(), CostCount(0.5), TcCap(6), AdditiveBonus(),
       UserCap(1), UserCap(2), UserCap(3), UserCap(4), UserCap(6), UserCap(3, True), UserCap(4, True),
       SpecMalus(0.5), CoreCap(0.25, "heavy"), CoreCap(0.10, "test"),
       # second round: finer steps along the two best families
       CostCount(0.3), CostCount(0.4), CostCount(0.65), CostCountAdd(0.5), CostCountAdd(0.3),
       UserCap(8, True), UserCap(12, True), TcCap(16)]
BY_NAME = {p.name: p for p in ALL}
