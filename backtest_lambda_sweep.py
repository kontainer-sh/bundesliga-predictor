#!/usr/bin/env python3
"""λ-Sweep: optimaler Mischfaktor zwischen DC-Modell und Quoten.

Walk-forward Backtest auf Saison 2024/2025 BL1, sweept ODDS_WEIGHT
von 0.0 (nur Modell) bis 1.0 (nur Quoten) in 0.1-Schritten.
Pro Spieltag wird DC nur einmal gefittet — günstig.
"""
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt


TEST_SEASONS = [2022, 2023, 2024]
LAMBDAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]


def sweep_season(test_season: int):
    """Walk-Forward Sweep für eine Saison. Gibt {lambda: [pts pro Spiel]} zurück."""
    print(f"\n--- Saison {test_season}/{test_season+1} ---")
    all_matches = kt.load_all_matches(test_season)
    season_matches = [m for m in all_matches
                      if m["league"] == "bl1" and m["season"] == test_season]

    odds_data = {}
    odds_rows = kt.fetch_odds_csv(test_season)
    for row in odds_rows:
        key = (kt._normalize_team(row["home"]), kt._normalize_team(row["away"]))
        odds_data[key] = {k: v for k, v in row.items()}
    print(f"  {len(season_matches)} Spiele, {len(odds_data)} mit Quoten")

    pts = {lam: [] for lam in LAMBDAS}

    for md in range(1, 35):
        cutoff = [m for m in season_matches if m["matchday"] < md]
        prev = [m for m in all_matches
                if not (m["league"] == "bl1" and m["season"] == test_season)]
        training = prev + cutoff
        if len(training) < kt.MIN_MATCHES:
            continue
        md_matches = [m for m in season_matches if m["matchday"] == md]
        if not md_matches:
            continue

        ref_date = min(m["date"] for m in md_matches)
        model = kt.fit_dixon_coles(training, ref_date)

        for m in md_matches:
            home, away = m["home"], m["away"]
            if home not in model["attack"] or away not in model["attack"]:
                continue
            real_h, real_a = m["home_goals"], m["away_goals"]

            od = kt._find_odds(odds_data, kt._normalize_team(home),
                               kt._normalize_team(away)) if odds_data else None
            dc_mat = kt.score_matrix(home, away, model)
            o_mat = kt.odds_to_score_matrix(
                od["p_home"], od["p_draw"], od["p_away"],
                od.get("p_over"), od.get("ou_line", 2.5)) if od else None

            for lam in LAMBDAS:
                if o_mat is None:
                    mat = dc_mat
                else:
                    mat = (1 - lam) * dc_mat + lam * o_mat
                    mat /= mat.sum()
                ev = np.einsum("ra,tpra->tp", mat, kt._POINTS_TABLE)
                ev_clipped = ev[:kt.MAX_TIP_GOALS + 1, :kt.MAX_TIP_GOALS + 1]
                idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
                pts[lam].append(kt.kicktipp_points(idx[0], idx[1], real_h, real_a))
        if md % 5 == 0:
            print(f"  MD {md} fertig")

    return pts


def main():
    t_start = time.time()

    all_pts = {lam: [] for lam in LAMBDAS}
    per_season = {}
    for s in TEST_SEASONS:
        season_pts = sweep_season(s)
        per_season[s] = season_pts
        for lam in LAMBDAS:
            all_pts[lam].extend(season_pts[lam])

    print(f"\n{'='*72}")
    print(f"λ-SWEEP — Aggregat über Saisons {TEST_SEASONS}")
    print(f"{'='*72}")
    n_total = len(all_pts[0.7])
    print(f"Spiele insgesamt: {n_total}\n")

    header = f"{'λ':>4}  {'Σ Pkt':>7}  {'Ø/Spiel':>9}  {'Δ vs 0.7':>10}  | "
    header += "  ".join(f"S{s%100:02d}" for s in TEST_SEASONS)
    print(header)
    print("─" * len(header))

    baseline = sum(all_pts[0.7])
    for lam in LAMBDAS:
        total = sum(all_pts[lam])
        mean = total / n_total
        delta = total - baseline
        per = "  ".join(f"{sum(per_season[s][lam]):3d}" for s in TEST_SEASONS)
        marker = "  ← Produktion" if lam == 0.7 else ""
        print(f"{lam:.1f}  {total:7d}  {mean:9.3f}  {delta:+10d}  | {per}{marker}")

    best_lam = max(LAMBDAS, key=lambda l: sum(all_pts[l]))
    best_total = sum(all_pts[best_lam])
    print(f"\nBest: λ={best_lam}  Σ={best_total}  Δ vs 0.7: {best_total - baseline:+d}")

    if best_lam != 0.7:
        diffs = np.array(all_pts[best_lam]) - np.array(all_pts[0.7])
        rng = np.random.default_rng(42)
        boot_mean = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                              for _ in range(10000)])
        ci_lo, ci_hi = np.percentile(boot_mean, [2.5, 97.5])
        p_two = 2 * min((boot_mean <= 0).mean(), (boot_mean >= 0).mean())
        print(f"\nGepaarte Diff (best - 0.7) pro Spiel:")
        print(f"  Mittelwert: {diffs.mean():+.4f}")
        print(f"  95%-CI:     [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  Bootstrap p-value: {p_two:.3f}")

    print(f"\n--- Bestes λ pro Saison (Robustheits-Check) ---")
    for s in TEST_SEASONS:
        best_s = max(LAMBDAS, key=lambda l: sum(per_season[s][l]))
        print(f"  Saison {s}/{s+1}: λ={best_s}  Σ={sum(per_season[s][best_s])}")

    print(f"\nLaufzeit: {(time.time()-t_start)/60:.1f} Min")


if __name__ == "__main__":
    main()
