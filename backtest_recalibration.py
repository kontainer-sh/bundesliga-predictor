#!/usr/bin/env python3
"""Walk-Forward Backtest: Score-Matrix-Recalibration vs. Baseline.

Misst Kicktipp-Punkte pro Spieltag für Saison 2024/2025 BL1, mit:
- Baseline:   DC-Modell + football-data.co.uk Quoten (70/30)
- Recal:      Baseline + multiplikative Score-Cell-Korrektur aus
              fit_recalibration (trainiert auf 2022/2023 + 2023/2024)

Ohne Data-Leakage: Recal-Training nutzt KEINE Daten aus der Test-Saison.
"""
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt


TEST_SEASON = 2024
RECAL_TRAIN_SEASONS = [2022, 2023]


def main():
    t_start = time.time()

    print(f"Lade alle Matches (Test-Saison {TEST_SEASON}/{TEST_SEASON+1})...")
    all_matches = kt.load_all_matches(TEST_SEASON)
    season_matches = [m for m in all_matches
                      if m["league"] == "bl1" and m["season"] == TEST_SEASON]
    print(f"  {len(all_matches)} total, {len(season_matches)} in Test-Saison")

    print(f"\nLade Quoten Saison {TEST_SEASON}/{TEST_SEASON+1}...")
    odds_data = {}
    try:
        odds_rows = kt.fetch_odds_csv(TEST_SEASON)
        for row in odds_rows:
            key = (kt._normalize_team(row["home"]), kt._normalize_team(row["away"]))
            odds_data[key] = {k: v for k, v in row.items()}
        print(f"  {len(odds_data)} Spiele mit Quoten")
    except Exception as e:
        print(f"  Warnung: keine Quoten ({e})")

    print(f"\nLerne Recalibration aus Saisons {RECAL_TRAIN_SEASONS}...")
    t0 = time.time()
    correction = kt.fit_recalibration(RECAL_TRAIN_SEASONS, min_obs=20)
    print(f"  fertig in {time.time()-t0:.0f}s")
    print(f"  Korrektur-Range: {correction.min():.3f} – {correction.max():.3f}")
    print(f"  Cells != 1.0: {(correction != 1.0).sum()}/{correction.size}")

    print(f"\nWalk-Forward Backtest über Spieltage 1..34...")
    pts_baseline = []
    pts_recal = []
    diffs = []  # spielweise Differenz
    matches_used = 0
    matches_no_odds = 0

    for md in range(1, 35):
        cutoff = [m for m in season_matches if m["matchday"] < md]
        prev = [m for m in all_matches
                if not (m["league"] == "bl1" and m["season"] == TEST_SEASON)]
        training = prev + cutoff
        if len(training) < kt.MIN_MATCHES:
            continue

        md_matches = [m for m in season_matches if m["matchday"] == md]
        if not md_matches:
            continue

        ref_date = min(m["date"] for m in md_matches)
        t0 = time.time()
        model = kt.fit_dixon_coles(training, ref_date)

        md_pts_b = 0
        md_pts_r = 0
        for m in md_matches:
            home, away = m["home"], m["away"]
            if home not in model["attack"] or away not in model["attack"]:
                continue

            real_h, real_a = m["home_goals"], m["away_goals"]

            od = kt._find_odds(odds_data, kt._normalize_team(home),
                               kt._normalize_team(away)) if odds_data else None
            if not od:
                matches_no_odds += 1

            th_b, ta_b, _ = kt.compute_tip(home, away, model,
                                           {(kt._normalize_team(home), kt._normalize_team(away)): od} if od else None,
                                           correction_table=None)
            th_r, ta_r, _ = kt.compute_tip(home, away, model,
                                           {(kt._normalize_team(home), kt._normalize_team(away)): od} if od else None,
                                           correction_table=correction)

            p_b = kt.kicktipp_points(th_b, ta_b, real_h, real_a)
            p_r = kt.kicktipp_points(th_r, ta_r, real_h, real_a)

            pts_baseline.append(p_b)
            pts_recal.append(p_r)
            diffs.append(p_r - p_b)
            md_pts_b += p_b
            md_pts_r += p_r
            matches_used += 1

        delta = md_pts_r - md_pts_b
        sign = "+" if delta >= 0 else ""
        print(f"  MD {md:2d}: baseline={md_pts_b:3d}  recal={md_pts_r:3d}  Δ={sign}{delta:+d}  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"ERGEBNIS — Saison {TEST_SEASON}/{TEST_SEASON+1} BL1")
    print(f"{'='*60}")
    print(f"Spiele:                    {matches_used}")
    print(f"davon ohne Quoten:         {matches_no_odds}")
    sum_b = sum(pts_baseline)
    sum_r = sum(pts_recal)
    print(f"\nGesamtpunkte Baseline:     {sum_b}")
    print(f"Gesamtpunkte Recalibration:{sum_r}")
    print(f"Δ:                         {sum_r - sum_b:+d}  ({(sum_r-sum_b)/sum_b*100:+.1f}%)")

    mean_b = np.mean(pts_baseline)
    mean_r = np.mean(pts_recal)
    print(f"\nØ Punkte/Spiel Baseline:   {mean_b:.3f}")
    print(f"Ø Punkte/Spiel Recal:      {mean_r:.3f}")
    print(f"Δ Ø/Spiel:                 {mean_r - mean_b:+.4f}")

    diffs_arr = np.array(diffs)
    n = len(diffs_arr)
    se = diffs_arr.std(ddof=1) / np.sqrt(n)
    print(f"\nGepaarte Diff (Recal - Baseline):")
    print(f"  Mittelwert:   {diffs_arr.mean():+.4f}")
    print(f"  Std-Fehler:   {se:.4f}")
    print(f"  t-Statistik:  {diffs_arr.mean()/se:+.2f}")
    print(f"  95%-CI Mean:  [{diffs_arr.mean()-1.96*se:+.4f}, {diffs_arr.mean()+1.96*se:+.4f}]")

    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(diffs_arr, size=n, replace=True).sum() for _ in range(10000)])
    p_two_sided = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"  Paired Bootstrap p-value (10k): {p_two_sided:.3f}")

    print(f"\nLaufzeit: {(time.time()-t_start)/60:.1f} Min")


if __name__ == "__main__":
    main()
