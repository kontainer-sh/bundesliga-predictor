#!/usr/bin/env python3
"""Diebold-Mariano-Test: schlägt Pinnacle-Closing das DC-Modell auf RPS?

Motivation: Kicktipp-Punkte sind ein *unpropres* Scoring — sie belohnen exakte
Ergebnisse, deren Form Modell UND Quoten aus derselben Maschinerie
(odds_to_score_matrix) ziehen, und verwischen so den echten 1X2-Prognose-
Vorsprung des Marktes. Der **Ranked Probability Score (RPS)** ist ein propres
Scoring für geordnetes 1X2 (Constantinou & Fenton 2012) und misst die
Prognosegüte direkt.

Der **Diebold-Mariano-Test** (1995) prüft, ob die mittlere RPS-Differenz zweier
Forecasts signifikant von 0 abweicht. Die Verlustdifferenzen sind NICHT strikt
unabhängig: Spiele teilen sich Teams, und Team-Schätzfehler wirken über
benachbarte Spieltage. Daher wird die Signifikanz *dependence-aware* geprüft —
Newey-West-HAC-Langfristvarianz (Lag ~1 Spieltag) plus Moving-Block-Bootstrap —
und dem naiven iid-Lag-0 gegenübergestellt (Review-Finding 2026-08-21).

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
            model = kt.fit_dixon_coles(kt.training_split(allm, season, md, ref_date=ref), ref)
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


def dm_test(d: np.ndarray, lag: int = 0):
    """Diebold-Mariano auf der Verlustdifferenz d.

    lag=0 → iid-Varianz; lag>0 → Newey-West-HAC-Langfristvarianz (Bartlett-Kernel),
    die serielle Abhängigkeit (geteilte Teams / Nachbar-Spieltage) berücksichtigt.
    """
    n = len(d)
    dm = d - d.mean()
    lrv = (dm @ dm) / n  # γ0
    for k in range(1, lag + 1):
        gk = (dm[k:] @ dm[:-k]) / n
        lrv += 2 * (1 - k / (lag + 1)) * gk  # Bartlett-Gewicht
    var = lrv / n
    stat = d.mean() / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p


def block_bootstrap_ci(d: np.ndarray, block: int, n_boot: int, rng):
    """Moving-Block-Bootstrap für d̄ — erhält serielle Abhängigkeit (dependence-aware).

    Gibt (lo, hi, p) zurück; p ist der zweiseitige Bootstrap-p-Wert gegen d̄=0.
    """
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return lo, hi, p


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
    rng = np.random.default_rng(SEED)
    LAG = 9  # ~ ein Spieltag: fängt geteilte-Teams-/Nachbar-Spieltag-Abhängigkeit
    stat0, p0 = dm_test(d, lag=0)
    statL, pL = dm_test(d, lag=LAG)
    boot_iid = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo_i, hi_i = np.percentile(boot_iid, [2.5, 97.5])
    lo_b, hi_b, p_b = block_bootstrap_ci(d, block=LAG, n_boot=N_BOOT, rng=rng)

    print("\n── Diebold-Mariano: Modell vs. Closing (d = RPS_Modell − RPS_Closing) ──")
    print(f"  Mittlere Differenz d̄:      {d.mean():+.5f}  (>0 ⇒ Modell schlechter)")
    print(f"  DM iid (Lag 0):            stat {stat0:+.3f}, p={p0:.4f}  (naiv)")
    print(f"  DM Newey-West (Lag {LAG}):     stat {statL:+.3f}, p={pL:.4f}  (dependence-aware)")
    print(f"  iid-Bootstrap 95%-CI:      [{lo_i:+.5f}, {hi_i:+.5f}]")
    print(f"  Block-Bootstrap 95%-CI:    [{lo_b:+.5f}, {hi_b:+.5f}], p={p_b:.4f}  (Block={LAG})")
    sig = "SIGNIFIKANT" if pL < 0.05 else "NICHT signifikant"
    verdict = "Closing besser" if d.mean() > 0 else "Modell besser"
    print(f"\n  → Dependence-aware ({sig}, α=0.05). Punktschätzung: {verdict}.")
    print("     Richtung unabhängig repliziert (Pitcan 2026, Serie A, DC-Pooling-Gewicht 0.000).")


if __name__ == "__main__":
    main()
