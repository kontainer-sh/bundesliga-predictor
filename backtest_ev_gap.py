#!/usr/bin/env python3
"""EV-Gap-Sensitivität: wirken Modell-Effekte nur bei knappen Entscheidungen?

Für jedes Spiel:
- EV-Gap = EV(bester Tipp) - EV(zweitbester Tipp) in der Production-Score-Matrix (λ=0.7)
- Spiele in Gap-Bins einteilen
- Pro Bin: Disagreement-Rate Modell vs. Quoten, Punktedifferenz

Hypothese: Wenn das Modell echten Edge gegenüber reinen Quoten hat,
sollte er sich in den schmalen EV-Bins konzentrieren (knappe Entscheidungen).
Wenn der Edge dort fehlt, hat das Modell praktisch keinen Hebel — alle
zuversichtlichen Tipps sind ohnehin identisch zur Markt-Tendenz.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt


TEST_SEASONS = [2022, 2023, 2024]
LAMBDA_MODEL = 0.7
LAMBDA_ODDS = 1.0

# EV-Gap-Bins (Pkt/Spiel). Standard-Punkteschema: bis 3 Pkt pro Spiel.
GAP_BINS = [(0.000, 0.005), (0.005, 0.010), (0.010, 0.020),
            (0.020, 0.040), (0.040, 0.100)]


def tip_and_gap(mat: np.ndarray) -> tuple[int, int, float]:
    ev = np.einsum("ra,tpra->tp", mat, kt._POINTS_TABLE)
    ev_clipped = ev[:kt.MAX_TIP_GOALS + 1, :kt.MAX_TIP_GOALS + 1]
    flat = ev_clipped.flatten()
    order = np.argsort(flat)[::-1]
    best_idx = order[0]
    second_idx = order[1]
    best_h, best_a = np.unravel_index(best_idx, ev_clipped.shape)
    return int(best_h), int(best_a), float(flat[best_idx] - flat[second_idx])


def run_season(test_season: int):
    print(f"--- Saison {test_season}/{test_season+1} ---", flush=True)
    all_matches = kt.load_all_matches(test_season)
    season_matches = [m for m in all_matches
                      if m["league"] == "bl1" and m["season"] == test_season]

    odds_data = {}
    for row in kt.fetch_odds_csv(test_season):
        key = (kt._normalize_team(row["home"]), kt._normalize_team(row["away"]))
        odds_data[key] = {k: v for k, v in row.items()}

    rows = []
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
            od = kt._find_odds(odds_data, kt._normalize_team(home),
                               kt._normalize_team(away))
            if od is None:
                continue
            dc_mat = kt.score_matrix(home, away, model)
            o_mat = kt.odds_to_score_matrix(
                od["p_home"], od["p_draw"], od["p_away"],
                od.get("p_over"), od.get("ou_line", 2.5))

            mat_m = (1 - LAMBDA_MODEL) * dc_mat + LAMBDA_MODEL * o_mat
            mat_m /= mat_m.sum()
            mat_o = o_mat / o_mat.sum()

            tm_h, tm_a, gap_m = tip_and_gap(mat_m)
            to_h, to_a, _ = tip_and_gap(mat_o)
            real_h, real_a = m["home_goals"], m["away_goals"]

            rows.append({
                "season": test_season,
                "gap_model": gap_m,
                "tip_model": (tm_h, tm_a),
                "tip_odds": (to_h, to_a),
                "pts_model": kt.kicktipp_points(tm_h, tm_a, real_h, real_a),
                "pts_odds": kt.kicktipp_points(to_h, to_a, real_h, real_a),
                "agree": (tm_h, tm_a) == (to_h, to_a),
            })
    return rows


def bin_label(lo, hi):
    return f"[{lo:.3f}, {hi:.3f})"


def main():
    t_start = time.time()
    all_rows = []
    for s in TEST_SEASONS:
        all_rows.extend(run_season(s))

    n = len(all_rows)
    print(f"\n{'='*82}")
    print(f"EV-GAP-SENSITIVITÄT — Modell (λ={LAMBDA_MODEL}) vs. Quoten (λ={LAMBDA_ODDS})")
    print(f"Aggregat über Saisons {TEST_SEASONS}, {n} Spiele")
    print(f"{'='*82}\n")

    # Verteilung der Gaps
    gaps = np.array([r["gap_model"] for r in all_rows])
    print(f"EV-Gap-Verteilung: min={gaps.min():.3f} med={np.median(gaps):.3f} "
          f"max={gaps.max():.3f}\n")

    # Pro Bin
    print(f"{'Bin (EV-Gap)':<16} {'N':>4} {'Disagree':>9} {'Modell':>7} {'Quoten':>7} "
          f"{'Δ':>6} {'Δ/Spiel':>9}")
    print("─" * 82)

    for lo, hi in GAP_BINS:
        bin_rows = [r for r in all_rows if lo <= r["gap_model"] < hi]
        if not bin_rows:
            print(f"{bin_label(lo, hi):<16} {'-':>4}")
            continue
        n_bin = len(bin_rows)
        dis = sum(1 for r in bin_rows if not r["agree"])
        pm = sum(r["pts_model"] for r in bin_rows)
        po = sum(r["pts_odds"] for r in bin_rows)
        delta = pm - po
        per = delta / n_bin
        print(f"{bin_label(lo, hi):<16} {n_bin:>4} "
              f"{dis:>5} ({100*dis/n_bin:4.1f}%) {pm:>7} {po:>7} {delta:>+6} {per:>+9.4f}")

    # Bootstrap auf jedem Bin
    print(f"\nPaired Bootstrap (n=10000) pro Bin — Δ-CI und p-value:")
    print(f"{'Bin':<16} {'N':>4} {'Δ (Pkt/Spiel)':>14} {'95%-CI':>22} {'p':>7}")
    print("─" * 82)
    rng = np.random.default_rng(42)
    for lo, hi in GAP_BINS:
        bin_rows = [r for r in all_rows if lo <= r["gap_model"] < hi]
        if len(bin_rows) < 10:
            continue
        diffs = np.array([r["pts_model"] - r["pts_odds"] for r in bin_rows])
        boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                         for _ in range(10000)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        print(f"{bin_label(lo, hi):<16} {len(bin_rows):>4} "
              f"{diffs.mean():>+14.4f} "
              f"[{ci_lo:>+7.4f}, {ci_hi:>+7.4f}] {p_two:>7.3f}")

    # Disagreement nur in schmalen Bins?
    print(f"\nDisagreement-Konzentration:")
    total_dis = sum(1 for r in all_rows if not r["agree"])
    cumulative = 0
    for lo, hi in GAP_BINS:
        bin_dis = sum(1 for r in all_rows
                      if lo <= r["gap_model"] < hi and not r["agree"])
        cumulative += bin_dis
        share = 100 * bin_dis / total_dis if total_dis else 0
        cum_share = 100 * cumulative / total_dis if total_dis else 0
        print(f"  {bin_label(lo, hi):<16} {bin_dis:>4} Disagree "
              f"({share:4.1f}% des Disagree-Pools, kum. {cum_share:4.1f}%)")

    print(f"\nLaufzeit: {(time.time()-t_start)/60:.1f} Min")


if __name__ == "__main__":
    main()
