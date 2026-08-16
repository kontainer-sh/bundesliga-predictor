#!/usr/bin/env python3
"""Diebold-Mariano-Test: schlägt Pinnacle-Closing das DC-Modell auf RPS?

Motivation: Kicktipp-Punkte sind ein *unpropres* Scoring — sie belohnen exakte
Ergebnisse, deren Form Modell UND Quoten aus derselben Maschinerie
(odds_to_score_matrix) ziehen, und verwischen so den echten 1X2-Prognose-
Vorsprung des Marktes. Der **Ranked Probability Score (RPS)** ist ein propres
Scoring für geordnetes 1X2 (Constantinou & Fenton 2012) und misst die
Prognosegüte direkt.

Der **Diebold-Mariano-Test** (1995) prüft, ob die mittlere RPS-Differenz zweier
Forecasts signifikant von 0 abweicht. Spiele sind über die Saison praktisch
unkorreliert (keine Überlappung der Prognose-Horizonte) → DM mit Lag 0. Ein
Paired Bootstrap dient als verteilungsfreier Cross-Check.

Konvention: d = RPS(Modell) − RPS(Closing). d > 0 ⇒ Modell hat den *höheren*
(schlechteren) RPS ⇒ Closing ist besser.

Reproduktion: `python backtest_dm_test.py`
"""
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
import kicktipp as kt

TEST_SEASONS = [2022, 2023, 2024, 2025]
N_BOOT = 10000
SEED = 42


def rps_1x2(probs, rh, ra) -> float:
    """Ranked Probability Score für geordnetes 1X2 (Heim, Remis, Auswärts)."""
    o = (1, 0, 0) if rh > ra else ((0, 1, 0) if rh == ra else (0, 0, 1))
    cp = co = s = 0.0
    for i in range(2):  # r-1 = 2 kumulative Terme
        cp += probs[i]
        co += o[i]
        s += (cp - co) ** 2
    return s / 2.0


def model_1x2(mat: np.ndarray):
    """1X2-Randwahrscheinlichkeiten aus einer Score-Matrix."""
    return np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()


def collect():
    """Sammelt paarweise (RPS_Modell, RPS_Closing, RPS_kombiniert) je Spiel."""
    rows = {"model": [], "odds": [], "comb": [], "season": []}
    for season in TEST_SEASONS:
        allm = kt.load_all_matches(season)
        sb = [m for m in allm if m["league"] == "bl1" and m["season"] == season]
        odds = {(kt._normalize_team(r["home"]), kt._normalize_team(r["away"])): r
                for r in kt.fetch_odds_csv(season)}  # Default: Closing
        for md in range(1, max(m["matchday"] for m in sb) + 1):
            mdm = [m for m in sb if m["matchday"] == md]
            if not mdm:
                continue
            ref = min(m["date"] for m in mdm)
            model = kt.fit_dixon_coles(kt.training_split(allm, season, md), ref)
            for m in mdm:
                h, a, rh, ra = m["home"], m["away"], m["home_goals"], m["away_goals"]
                if h not in model["attack"] or a not in model["attack"]:
                    continue
                od = kt._find_odds(odds, h, a)
                if not od:
                    continue
                dc = kt.score_matrix(h, a, model)
                omat = kt.odds_to_score_matrix(od["p_home"], od["p_draw"], od["p_away"])
                comb = (1 - kt.ODDS_WEIGHT) * dc + kt.ODDS_WEIGHT * omat
                comb /= comb.sum()
                rows["model"].append(rps_1x2(model_1x2(dc), rh, ra))
                rows["odds"].append(rps_1x2((od["p_home"], od["p_draw"], od["p_away"]), rh, ra))
                rows["comb"].append(rps_1x2(model_1x2(comb), rh, ra))
                rows["season"].append(season)
    return {k: np.array(v) for k, v in rows.items()}


def dm_test(d: np.ndarray):
    """Diebold-Mariano (Lag 0) auf der Verlustdifferenz d. Gibt (stat, p)."""
    n = len(d)
    var = d.var(ddof=1) / n
    stat = d.mean() / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p


def main():
    data = collect()
    n = len(data["model"])
    print(f"\n{n} gewertete Spiele über Saisons {TEST_SEASONS}\n")

    print(f"{'Metrik':16}{'RPS Ø':>9}  (niedriger = besser)")
    for key, label in (("model", "Modell (DC)"), ("odds", "Pinnacle Closing"),
                       ("comb", "Kombiniert λ=0.7")):
        print(f"  {label:14}{data[key].mean():>9.4f}")

    print("\nPer Saison (RPS Ø — Modell / Closing):")
    for s in TEST_SEASONS:
        mask = data["season"] == s
        mm, oo = data["model"][mask].mean(), data["odds"][mask].mean()
        flag = "  Modell besser" if mm < oo else ""
        print(f"  {s}/{str(s+1)[2:]}: {mm:.4f} / {oo:.4f}{flag}")

    d = data["model"] - data["odds"]
    stat, p = dm_test(d)
    rng = np.random.default_rng(SEED)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("\n── Diebold-Mariano: Modell vs. Closing (d = RPS_Modell − RPS_Closing) ──")
    print(f"  Mittlere Differenz d̄:   {d.mean():+.5f}  (>0 ⇒ Modell schlechter)")
    print(f"  DM-Statistik:           {stat:+.3f}")
    print(f"  p-Wert (zweiseitig):    {p:.4f}")
    print(f"  Paired-Bootstrap 95%-CI: [{lo:+.5f}, {hi:+.5f}]  (n={N_BOOT})")
    sig = "SIGNIFIKANT" if p < 0.05 else "NICHT signifikant"
    verdict = ("Closing besser" if d.mean() > 0 else "Modell besser")
    print(f"\n  → {sig} (α=0.05). Punktschätzung: {verdict}, aber "
          f"{'der Unterschied ist real' if p < 0.05 else 'im Rahmen des Rauschens'}.")


if __name__ == "__main__":
    main()
