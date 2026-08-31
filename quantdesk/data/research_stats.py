"""Screening statistics that keep a study honest.

Extracted so every screen counts its tests the same way. The 1h gold screen
had a version of this that reset its counter between processes and printed
"clears correction" for a t that cleared nothing -- the fix belongs in one
place rather than in each script that happens to remember.

Three things this enforces:

* Multiple testing is counted across the whole screen, not per section.
  Twenty tests at p<0.05 produce one hit from pure noise by construction.
* Every effect is reported against a COST FLOOR. A statistically real
  effect smaller than the round-trip cost is not a trade, and reporting
  significance without that comparison invites the mistake.
* Every number carries its sample size and its confidence interval. A
  point estimate travelling alone is how a 12-observation bucket ends up
  quoted as a finding.
"""

from __future__ import annotations

import math
import statistics as st


def inv_norm(p: float) -> float:
    """Inverse standard normal CDF (Acklam). No scipy dependency."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def stats(xs: list[float]) -> dict | None:
    n = len(xs)
    if n < 2:
        return None
    mean = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": mean / se if se else float("nan"),
            "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


def welch(a: list[float], b: list[float]) -> dict | None:
    """Difference of means with unequal variances. Two regimes are not two
    samples from one distribution, so pooling their variance would be the
    wrong test and would usually be the more flattering one."""
    sa, sb = stats(a), stats(b)
    if sa is None or sb is None:
        return None
    se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
    diff = sa["mean"] - sb["mean"]
    return {"diff": diff, "se": se, "t": diff / se if se else float("nan"),
            "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "n_a": sa["n"], "n_b": sb["n"]}


def overlap_corrected(xs: list[float], horizon: int) -> dict | None:
    """Stats for OVERLAPPING windows, with an honest standard error.

    Non-overlapping sampling is safe but wasteful: at a 63-day horizon it
    turns 3,298 sessions into 52 observations, and the confidence interval
    then only excludes effects nobody claims exist.

    Overlapping windows use every session, which estimates the MEAN far
    better. They do not give more independent information, so the standard
    error must be widened by sqrt(horizon) -- adjacent windows share
    horizon-1 periods, and dividing by sqrt(n) instead of sqrt(n/horizon)
    is precisely the error that makes noise look significant.

    This is the same correction the VRP study applies, restated here so a
    screen can use it without importing the study.
    """
    s = stats(xs)
    if s is None:
        return None
    n_indep = max(len(xs) / float(horizon), 1.0)
    se = s["sd"] / math.sqrt(n_indep)
    return {"n": s["n"], "n_indep": n_indep, "mean": s["mean"], "sd": s["sd"],
            "se": se, "t": s["mean"] / se if se else float("nan"),
            "lo": s["mean"] - 1.96 * se, "hi": s["mean"] + 1.96 * se,
            "naive_t": s["t"]}


def welch_overlap(a: list[float], b: list[float], horizon: int) -> dict | None:
    """Difference of two overlapping-window means, both SEs corrected."""
    sa = overlap_corrected(a, horizon)
    sb = overlap_corrected(b, horizon)
    if sa is None or sb is None:
        return None
    se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
    diff = sa["mean"] - sb["mean"]
    return {"diff": diff, "se": se, "t": diff / se if se else float("nan"),
            "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "n_a": sa["n"], "n_b": sb["n"],
            "indep_a": sa["n_indep"], "indep_b": sb["n_indep"]}


class Screen:
    """Counts every test in one screen and corrects across all of them."""

    def __init__(self, alpha: float = 0.05, cost_bp: float = 0.0):
        self.alpha = alpha
        self.cost_bp = cost_bp
        self.results: list[tuple[str, dict, str, bool, bool]] = []

    @property
    def n_tests(self) -> int:
        return len(self.results)

    def record(self, label: str, xs: list[float], unit: str = "bp",
               costed: bool = True, descriptive: bool = False) -> dict | None:
        """`descriptive=True` for a bucket's LEVEL rather than a hypothesis.

        The GEX screen made this necessary. It recorded each regime's mean
        realised vol, and summary() then announced "short gamma, t=+24.71"
        as clearing Bonferroni -- a statement that realised volatility is
        greater than zero, which is true of every session ever traded and
        is not a finding.

        The hypothesis is always the DIFFERENCE between buckets. Levels are
        context. They still count toward the test total, because looking is
        looking, but they never appear as discoveries.
        """
        s = stats(xs)
        if s is None:
            print("  {0:<28} n={1} -- too few".format(label, len(xs)))
            self.results.append((label, {"n": len(xs), "t": 0.0}, unit,
                                costed, descriptive))
            return None
        self.results.append((label, s, unit, costed, descriptive))
        note = ""
        if costed and self.cost_bp and abs(s["mean"]) < self.cost_bp:
            note = "  (below the {0:.1f}bp cost floor)".format(self.cost_bp)
        print("  {0:<28} n={1:<5} mean {2:>+9.2f} {3:<8} t={4:>+6.2f}"
              "  95% CI [{5:>+8.2f}, {6:>+8.2f}]{7}".format(
                  label, s["n"], s["mean"], unit, s["t"],
                  s["lo"], s["hi"], note))
        return s

    def register_difference(self, label: str, w: dict, unit: str) -> None:
        """Record a between-bucket comparison as a real hypothesis test."""
        self.results.append((label, {"n": w["n_a"] + w["n_b"],
                                     "mean": w["diff"], "t": w["t"],
                                     "lo": w["lo"], "hi": w["hi"]},
                             unit, False, False))

    @property
    def n_hypotheses(self) -> int:
        return sum(1 for *_, desc in self.results if not desc)

    def threshold(self, tests: int | None = None) -> float:
        # Correct across HYPOTHESES, not across every number printed.
        # Descriptive levels are context; they are not claims.
        k = max(tests if tests is not None else self.n_hypotheses, 1)
        return inv_norm(1.0 - self.alpha / k / 2.0)

    def summary(self) -> None:
        thr = self.threshold()
        print("=" * 72)
        print("{0} hypothesis tests ({1} descriptive levels alongside). "
              "Bonferroni at alpha={2}: |t| > {3:.2f}".format(
                  self.n_hypotheses,
                  sum(1 for *_, d in self.results if d), self.alpha, thr))
        survivors = [(lbl, s, unit, costed) for lbl, s, unit, costed, desc
                     in self.results
                     if not desc and s.get("n", 0) >= 2 and abs(s["t"]) >= thr]
        if not survivors:
            print("Nothing clears the corrected threshold.")
        else:
            print("Clears correction:")
            for lbl, s, unit, costed in survivors:
                flag = ""
                if costed and self.cost_bp and abs(s["mean"]) < self.cost_bp:
                    flag = "  <-- but below the cost floor: not a trade"
                print("  {0:<28} mean {1:>+9.2f} {2:<8} t={3:>+6.2f}{4}"
                      .format(lbl, s["mean"], unit, s["t"], flag))
        nominal = sum(1 for _, s, _, _, desc in self.results
                      if not desc and s.get("n", 0) >= 2 and abs(s["t"]) >= 1.96)
        print("\n{0} would have passed an uncorrected 1.96. {1} tests on pure"
              " noise produce {2:.1f} such hits by construction."
              .format(nominal, self.n_tests, self.n_tests * self.alpha))


def describe_split(buckets: dict[str, list[float]], a: str, b: str,
                   unit: str = "bp", screen: "Screen | None" = None,
                   label: str | None = None) -> None:
    """The difference between two regimes, with its own error bar.

    Two buckets each failing to differ from zero can still differ from each
    other, and two buckets that each look significant can still be
    indistinguishable. The comparison people actually care about is this
    one, so it gets computed rather than eyeballed off two rows above it.
    """
    w = welch(buckets.get(a, []), buckets.get(b, []))
    if w is None:
        print("    {0} vs {1}: not enough observations".format(a, b))
        return
    print("    -> {0} minus {1}: {2:+.2f} {3}  t={4:+.2f}  "
          "95% CI [{5:+.2f}, {6:+.2f}]".format(
              a, b, w["diff"], unit, w["t"], w["lo"], w["hi"]))
    if screen is not None:
        # THE DIFFERENCE IS THE HYPOTHESIS. Registering it here is the
        # whole point: the screen was counting bucket LEVELS and never the
        # comparison anyone actually cared about, so the correction was
        # being applied to the wrong set of numbers entirely.
        screen.register_difference(label or "{0} - {1}".format(a, b),
                                   w, unit)
