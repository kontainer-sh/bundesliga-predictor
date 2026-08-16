#!/usr/bin/env python3
"""Headroom der Exakt-Ergebnis-Schicht: was könnten Correct-Score-Quoten bringen?

Kontext: ~30 % der Kicktipp-Punkte stammen aus exakten Ergebnissen (siehe
Punkte-Zerlegung 2026-08-16). Die exakte Ergebnisverteilung erzeugt aktuell
`odds_to_score_matrix` rein aus 1X2 + O/U-2.5 — der Markt fließt NICHT über die
gemeinsame Score-Verteilung ein. Correct-Score-Quoten würden genau hier ansetzen.

Diese Analyse beziffert die *Obergrenze* dieses Hebels OHNE CS-Daten. Der
Score-Layer kann die **Tendenz** nicht verbessern (Markt-Job); er kann nur
*innerhalb* der committeten Tendenz das Ergebnis schärfen. Gemessen wird eine
Leiter (Ø Kicktipp-Pkt/Spiel, gleiche Spiele):

  1. Naiv „immer 2:1"                     — Kontext (viele Casual-Tipper)
  2. Aktuell (Modell+Closing, EV-max)     — Baseline
  3. Beste konstante Scoreline / Tendenz  — populations-optimale CS-Info (Hindsight)
  4. Perfektes Ergebnis | Tendenz fix     — lose Score-Layer-Obergrenze
  5. Absolut (tatsächliches Ergebnis)     — triviale Decke (3.0)

Δ(3−2) = realistischer Headroom populations-typischer CS-Quoten.
Δ(4−2) = absolute Obergrenze der Score-Schicht bei fixer Tendenz.

Referenz: Die Score-Matrix-Recalibration (2026-05-10) hat populations-empirische
Score-Korrekturen bereits getestet → −27 bis −35 Pkt (Bias nicht saisonstabil).

Repro: `python backtest_score_headroom.py`
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt

TEST_SEASONS = [2022, 2023, 2024, 2025]

K = kt.MAX_TIP_GOALS + 1
CANDS = [(th, ta) for th in range(K) for ta in range(K)]


def tend(h, a):
    return int(np.sign(h - a))


def collect():
    """Sammelt je Spiel: Modell-Tipp, tatsächliches Ergebnis."""
    tips, results = [], []
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
            model = kt.fit_dixon_coles(kt.training_split(allm, season, md), ref)
            for m in mdm:
                h, a = m["home"], m["away"]
                if h not in model["attack"] or a not in model["attack"]:
                    continue
                if not kt._find_odds(odds, h, a):
                    continue
                th, ta, _ = kt.compute_tip(h, a, model, odds)
                tips.append((th, ta))
                results.append((m["home_goals"], m["away_goals"]))
    return tips, results


def main():
    tips, results = collect()
    n = len(tips)
    kp = kt.kicktipp_points

    # 1. Naiv immer 2:1
    naive = sum(kp(2, 1, rh, ra) for rh, ra in results)
    # 2. Baseline (aktuelle Tipps)
    base = sum(kp(th, ta, rh, ra) for (th, ta), (rh, ra) in zip(tips, results))
    # 4. Perfektes Ergebnis bei fixer (Modell-)Tendenz
    ceil_fix = sum(3 if tend(th, ta) == tend(rh, ra) else 0
                   for (th, ta), (rh, ra) in zip(tips, results))
    # 5. Absolut
    absolute = 3 * n

    # 3. Beste konstante Scoreline pro (Modell-)Tendenz — Hindsight
    const_total = 0
    for t in (1, 0, -1):
        idx = [i for i in range(n) if tend(*tips[i]) == t]
        if not idx:
            continue
        cands_t = [(h, a) for (h, a) in CANDS if tend(h, a) == t]
        best = max(sum(kp(ch, ca, *results[i]) for i in idx) for (ch, ca) in cands_t)
        const_total += best

    def line(label, total, tag=""):
        print(f"  {label:38}{total/n:6.3f}   ({total} Pkt){tag}")

    print(f"\n{n} Spiele über Saisons {TEST_SEASONS}\n")
    print(f"{'Strategie':40}{'Ø/Spiel':>7}")
    print("  " + "-" * 60)
    line("1. Naiv „immer 2:1\"", naive)
    line("2. Aktuell (Modell+Closing, EV-max)", base, "  ← Baseline")
    line("3. Beste konst. Scoreline / Tendenz", const_total, "  (CS-Proxy, Hindsight)")
    line("4. Perfektes Ergebnis | Tendenz fix", ceil_fix, "  (Score-Layer-Decke)")
    line("5. Absolut (tatsächliches Ergebnis)", absolute, "  (trivial)")

    print("\nHeadroom der Exakt-Ergebnis-Schicht:")
    print(f"  Δ(3−2) realistisch (populations-CS): {(const_total-base)/n:+.3f} Pkt/Spiel "
          f"({const_total-base:+d} Pkt/Saison-Set)")
    print(f"  Δ(4−2) absolute Obergrenze:          {(ceil_fix-base)/n:+.3f} Pkt/Spiel "
          f"({ceil_fix-base:+d} Pkt)")
    print("\n  Referenz: populations-empirische Score-Korrektur (Recal 2026-05-10)")
    print("  brachte −27 bis −35 Pkt → Δ(3) ist optimistisch (Hindsight, in-sample).")


if __name__ == "__main__":
    main()
