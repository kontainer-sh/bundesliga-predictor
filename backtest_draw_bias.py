#!/usr/bin/env python3
"""Remis-Bias: unterschätzt/untertippt das Modell Unentschieden?

Motivation: Im Punkteschema zahlt eine korrekte Remis-*Tendenz* 2 Punkte, eine
Sieg-Tendenz nur 1 — Remis sind also doppelt wertvoll. Dixon-Coles-Poisson
unterschätzt in der Literatur Unentschieden. Falls das Modell P(Remis) zu niedrig
ansetzt, tippt EV-max zu selten Remis und lässt Punkte liegen.

Test: (a) Kalibrierung — Ø P(Remis) vs. tatsächliche Remis-Quote. (b) Remis-Boost:
die Diagonale der Produktions-Score-Matrix mit δ skalieren, renormieren, EV-max neu
bestimmen und die Kicktipp-Punkte über mehrere Saisons vergleichen.

Ergebnis (2022–2025, 1186 Spiele): P(Remis)=0.236 vs. 0.250 real (kaum verzerrt);
δ=1.0 ist optimal, jeder Boost verschlechtert. EV-max wägt den 2-Punkte-Bonus
bereits korrekt ab → kein ausnutzbarer Remis-Edge. NULL-Resultat.

Repro: `python backtest_draw_bias.py`
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt

TEST_SEASONS = [2022, 2023, 2024, 2025]
DELTAS = [1.0, 1.15, 1.3, 1.5, 2.0]


def evmax(mat):
    ev = np.einsum("ra,tpra->tp", mat, kt._POINTS_TABLE)[:kt.MAX_TIP_GOALS + 1, :kt.MAX_TIP_GOALS + 1]
    i = np.unravel_index(ev.argmax(), ev.shape)
    return int(i[0]), int(i[1])


def main():
    diag = np.eye(kt.MAX_GOALS + 1, dtype=bool)
    tot = {d: 0 for d in DELTAS}
    n = pdraw_sum = actual_draw = model_draw_tips = 0

    for season in TEST_SEASONS:
        allm = kt.load_all_matches(season)
        sb = [m for m in allm if m["league"] == "bl1" and m["season"] == season]
        odds = {(kt._normalize_team(r["home"]), kt._normalize_team(r["away"])): r
                for r in kt.fetch_odds_csv(season)}
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
                n += 1
                dc = kt.score_matrix(h, a, model)
                om = kt.odds_to_score_matrix(od["p_home"], od["p_draw"], od["p_away"])
                base = (1 - kt.ODDS_WEIGHT) * dc + kt.ODDS_WEIGHT * om
                base /= base.sum()
                pdraw_sum += np.trace(base)
                actual_draw += (rh == ra)
                bt = evmax(base)
                model_draw_tips += (bt[0] == bt[1])
                for d in DELTAS:
                    m2 = base.copy()
                    m2[diag] *= d
                    m2 /= m2.sum()
                    th, ta = evmax(m2)
                    tot[d] += kt.kicktipp_points(th, ta, rh, ra)

    print(f"\n{n} Spiele über Saisons {TEST_SEASONS}\n")
    print(f"Modell Ø P(Remis): {pdraw_sum/n:.3f}   tatsächliche Remis-Quote: {actual_draw/n:.3f}")
    print(f"Modell tippt Remis in {model_draw_tips/n*100:.1f}% der Spiele (δ=1)\n")
    print(f"{'δ (Remis-Boost)':16}{'Σ Pkt':>8}{'Ø/Spiel':>9}{'Δ vs 1.0':>10}")
    for d in DELTAS:
        print(f"{d:<16}{tot[d]:>8}{tot[d]/n:>9.3f}{(tot[d]-tot[1.0])/n:>+10.3f}")
    print("\nBefund: δ=1.0 optimal → kein ausnutzbarer Remis-Edge (EV-max korrekt).")


if __name__ == "__main__":
    main()
