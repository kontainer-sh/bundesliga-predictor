#!/usr/bin/env python3
"""Definitiv-Test des Exact-Score-Proxys: Bringt eine gemeinsam auf 1X2 + O/U +
Asian Handicap gefittete Dixon-Coles-Score-Matrix (mit ρ) mehr Kicktipp-Punkte
als die 1X2-only-Rekonstruktion?

Drei gestaffelte Rekonstruktionen, jeweils EV-optimaler Kicktipp-Tipp, gegen das
reale Ergebnis gewertet — gepaart auf denselben Spielen:

  A  1X2-only            (Prod-Baseline: kt.odds_to_score_matrix)
  B  1X2 + O/U           (Prod-Hook: p_over)
  C  1X2 + O/U + AH      (neu: Dixon-Coles-Fit mit ρ auf alle drei Märkte)

A→B isoliert den O/U-Beitrag, B→C den novel AH-Beitrag. Erwartung laut Diagnose
(backtest_score_markets.py): AH trägt ~nichts über 1X2 → C ≈ B; O/U ist die
falsche Achse für Kicktipp → B ≲ A. Dieser Test liefert die *harte* Punktzahl.

Rein diagnostisch, kein Prod-Change. Offline aus dem CSV-Cache.
Aufruf:  python backtest_score_proxy.py [season_from] [season_to]   (Default 2019 2025)
"""
import math
import sys

import numpy as np
from scipy.optimize import minimize

import kicktipp as kt
from backtest_score_markets import load_rows, over_prob, margin_pmf, ah_model_prob

N = kt.MAX_GOALS + 1


def _pois(lam: float) -> np.ndarray:
    ks = np.arange(N)
    return np.exp(-lam) * lam**ks / np.array([math.factorial(k) for k in ks])


def dc_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    """Dixon-Coles-Score-Matrix: unabhängiges Poisson + Niedrig-Score-Korrektur τ(ρ)."""
    mat = np.outer(_pois(lh), _pois(la))
    tau = np.array([
        [1 - lh * la * rho, 1 + lh * rho],
        [1 + la * rho,      1 - rho],
    ])
    if np.any(tau <= 0):
        return None
    mat[:2, :2] *= tau
    s = mat.sum()
    return mat / s if s > 0 else None


def _tendency(mat):
    ph = np.sum(np.tril(mat, -1))
    pd = np.trace(mat)
    pa = np.sum(np.triu(mat, 1))
    return ph, pd, pa


def _kl2(p, q):
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = min(max(q, 1e-6), 1 - 1e-6)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def fit_proxy(p_home, p_draw, p_away, p_over, ah_line, p_ah_home):
    """Fittet (Total, Ratio, ρ) per Nelder-Mead auf 1X2 + O/U(2.5) + AH gemeinsam."""
    def objective(x):
        log_ratio, log_total, rho = x
        if abs(rho) > 0.2:
            return 1e6
        total = math.exp(log_total)
        ratio = math.exp(log_ratio)
        lh = total * ratio / (1 + ratio)
        la = total / (1 + ratio)
        mat = dc_matrix(lh, la, rho)
        if mat is None:
            return 1e6
        ph, pd, pa = _tendency(mat)
        tgt = np.array([p_home, p_draw, p_away])
        prd = np.clip(np.array([ph, pd, pa]), 1e-8, None)
        loss = float(np.sum(tgt * np.log(tgt / prd)))          # KL(1X2)
        loss += _kl2(p_over, over_prob(mat, 2.5))              # KL(O/U)
        p_mod, _ = ah_model_prob(margin_pmf(mat), ah_line)
        loss += _kl2(p_ah_home, p_mod)                         # KL(AH)
        return loss

    res = minimize(objective, [0.0, math.log(2.8), 0.0],
                   method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8})
    log_ratio, log_total, rho = res.x
    total, ratio = math.exp(log_total), math.exp(log_ratio)
    lh = total * ratio / (1 + ratio)
    la = total / (1 + ratio)
    return dc_matrix(lh, la, rho)


# EV-optimaler Kicktipp-Tipp (identisch zu compute_tip: argmax über 0..MAX_TIP_GOALS)
_PTS = np.array([[[[kt.kicktipp_points(th, ta, i, j) for j in range(N)] for i in range(N)]
                  for ta in range(kt.MAX_TIP_GOALS + 1)] for th in range(kt.MAX_TIP_GOALS + 1)])


def best_tip(mat):
    ev = np.einsum("ij,tpij->tp", mat, _PTS)
    th, ta = np.unravel_index(ev.argmax(), ev.shape)
    return int(th), int(ta)


def main():
    s_from = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    s_to = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    rows = []
    for s in range(s_from, s_to + 1):
        try:
            rows += load_rows(s)
        except Exception as e:
            print(f"  Warnung: Saison {s} nicht geladen ({e})")

    pA, pB, pC = [], [], []
    chg_ab = chg_bc = 0
    for r in rows:
        h = r["ah_line"]
        # nur Spiele mit vollem Closing + AH-Halb/Ganzlinie (fairer gepaarter Vergleich)
        if r["p_over_mkt"] is None or r["p_ah_home_mkt"] is None or h is None:
            continue
        if abs((2 * h) - round(2 * h)) > 1e-9:
            continue

        matA = kt.odds_to_score_matrix(r["p_home"], r["p_draw"], r["p_away"])
        matB = kt.odds_to_score_matrix(r["p_home"], r["p_draw"], r["p_away"],
                                       p_over=r["p_over_mkt"], ou_line=2.5)
        matC = fit_proxy(r["p_home"], r["p_draw"], r["p_away"],
                         r["p_over_mkt"], h, r["p_ah_home_mkt"])
        if matC is None:
            continue

        tA, tB, tC = best_tip(matA), best_tip(matB), best_tip(matC)
        hg, ag = r["hg"], r["ag"]
        pA.append(kt.kicktipp_points(*tA, hg, ag))
        pB.append(kt.kicktipp_points(*tB, hg, ag))
        pC.append(kt.kicktipp_points(*tC, hg, ag))
        chg_ab += (tA != tB)
        chg_bc += (tB != tC)

    pA, pB, pC = map(np.asarray, (pA, pB, pC))
    n = len(pA)
    print(f"\n=== EXACT-SCORE-PROXY Saisons {s_from}/{s_from+1}–{s_to}/{s_to+1} ===")
    print(f"  {n} Spiele mit vollem Closing (1X2+O/U+AH Halb/Ganzlinie)\n")
    print(f"  {'A  1X2-only':22} {pA.sum():5d} Pkt  ({pA.mean():.4f}/Spiel)")
    print(f"  {'B  1X2 + O/U':22} {pB.sum():5d} Pkt  ({pB.mean():.4f}/Spiel)")
    print(f"  {'C  1X2 + O/U + AH':22} {pC.sum():5d} Pkt  ({pC.mean():.4f}/Spiel)\n")
    print(f"  Tipp-Änderungen: A→B {chg_ab} ({chg_ab/n:.1%}),  B→C {chg_bc} ({chg_bc/n:.1%})\n")

    rng = np.random.default_rng(0)

    def boot(delta, label):
        b = [np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(5000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        sig = "signifikant" if (lo > 0 or hi < 0) else "n.s."
        print(f"  Δ {label}: {delta.sum():+d} Pkt ({delta.mean():+.4f}/Spiel), "
              f"95%-CI [{lo:+.4f}, {hi:+.4f}] → {sig}")

    boot(pB - pA, "B−A (O/U-Beitrag)")
    boot(pC - pB, "C−B (AH-Beitrag)")
    boot(pC - pA, "C−A (Proxy gesamt)")
    print("\nLesart: Ist C−A ≤ 0 bzw. n.s., bringt der Exact-Score-Proxy auf")
    print("Kicktipp-Punkten nichts — das Ceiling ist mit 1X2 bereits ausgeschöpft.")


if __name__ == "__main__":
    main()
