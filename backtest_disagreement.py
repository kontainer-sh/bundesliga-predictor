#!/usr/bin/env python3
"""Disagreement-Test: Modell (λ=0.7) vs. nur Quoten (λ=1.0).

Filtert auf Spiele, bei denen die beiden Strategien unterschiedlich tippen,
und vergleicht die Kicktipp-Punkte ausschließlich dort. Agreement-Spiele
liefern null Information, weil beide Strategien identische Punkte holen.

Frage: Hat das DC-Modell echten Edge gegenüber reiner Markt-Replikation,
oder ist der +10-Pkt-Vorteil aus dem λ-Sweep Rauschen?
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt


TEST_SEASONS = [2022, 2023, 2024]
LAMBDA_MODEL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
LAMBDA_ODDS = 1.0


def tip_for_lambda(dc_mat: np.ndarray, o_mat: np.ndarray, lam: float) -> tuple[int, int]:
    if o_mat is None:
        mat = dc_mat
    else:
        mat = (1 - lam) * dc_mat + lam * o_mat
        mat /= mat.sum()
    ev = np.einsum("ra,tpra->tp", mat, kt._POINTS_TABLE)
    ev_clipped = ev[:kt.MAX_TIP_GOALS + 1, :kt.MAX_TIP_GOALS + 1]
    idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
    return int(idx[0]), int(idx[1])


def run_season(test_season: int):
    print(f"\n--- Saison {test_season}/{test_season+1} ---")
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

            tm_h, tm_a = tip_for_lambda(dc_mat, o_mat, LAMBDA_MODEL)
            to_h, to_a = tip_for_lambda(dc_mat, o_mat, LAMBDA_ODDS)
            real_h, real_a = m["home_goals"], m["away_goals"]

            pts_m = kt.kicktipp_points(tm_h, tm_a, real_h, real_a)
            pts_o = kt.kicktipp_points(to_h, to_a, real_h, real_a)

            rows.append({
                "season": test_season,
                "md": md,
                "home": home, "away": away,
                "tip_model": (tm_h, tm_a),
                "tip_odds": (to_h, to_a),
                "real": (real_h, real_a),
                "pts_model": pts_m,
                "pts_odds": pts_o,
                "agree": (tm_h, tm_a) == (to_h, to_a),
            })
    print(f"  {len(rows)} Spiele mit Quoten ausgewertet")
    return rows


def main():
    t_start = time.time()
    all_rows = []
    for s in TEST_SEASONS:
        all_rows.extend(run_season(s))

    n_total = len(all_rows)
    agree = [r for r in all_rows if r["agree"]]
    disagree = [r for r in all_rows if not r["agree"]]

    print(f"\n{'='*72}")
    print(f"DISAGREEMENT-TEST — Modell (λ={LAMBDA_MODEL}) vs. Quoten (λ={LAMBDA_ODDS})")
    print(f"Aggregat über Saisons {TEST_SEASONS}")
    print(f"{'='*72}\n")
    print(f"Spiele gesamt:        {n_total}")
    print(f"Agreement (gleicher Tipp):    {len(agree):4d} ({100*len(agree)/n_total:4.1f}%)")
    print(f"Disagreement (versch. Tipp):  {len(disagree):4d} ({100*len(disagree)/n_total:4.1f}%)")

    pm_all = sum(r["pts_model"] for r in all_rows)
    po_all = sum(r["pts_odds"] for r in all_rows)
    pm_dis = sum(r["pts_model"] for r in disagree)
    po_dis = sum(r["pts_odds"] for r in disagree)

    print(f"\nGesamt (alle Spiele):")
    print(f"  Modell (λ={LAMBDA_MODEL}): {pm_all:4d} Pkt  (Ø {pm_all/n_total:.3f})")
    print(f"  Quoten (λ={LAMBDA_ODDS}): {po_all:4d} Pkt  (Ø {po_all/n_total:.3f})")
    print(f"  Δ Modell-Quoten: {pm_all-po_all:+d}")

    print(f"\nNur Disagreement-Spiele ({len(disagree)}):")
    print(f"  Modell: {pm_dis:4d} Pkt  (Ø {pm_dis/len(disagree):.3f})")
    print(f"  Quoten: {po_dis:4d} Pkt  (Ø {po_dis/len(disagree):.3f})")
    print(f"  Δ Modell-Quoten: {pm_dis-po_dis:+d}")

    # Paired bootstrap auf Disagreement-Spielen
    diffs = np.array([r["pts_model"] - r["pts_odds"] for r in disagree])
    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                           for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_two = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())

    print(f"\nPaired Bootstrap (n={n_boot}) auf Disagreement-Spielen:")
    print(f"  Mittelwert Δ:    {diffs.mean():+.4f} Pkt/Spiel")
    print(f"  95%-CI:          [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  p-value (zweiseitig): {p_two:.3f}")

    # Win-Loss-Tie auf Disagreement
    wins = (diffs > 0).sum()
    losses = (diffs < 0).sum()
    ties = (diffs == 0).sum()
    print(f"\nDisagreement-Outcome (pro Spiel):")
    print(f"  Modell besser:  {wins:4d} ({100*wins/len(disagree):.1f}%)")
    print(f"  Quoten besser:  {losses:4d} ({100*losses/len(disagree):.1f}%)")
    print(f"  Gleich:         {ties:4d} ({100*ties/len(disagree):.1f}%)")

    # Per-Saison auf Disagreement
    print(f"\nPro Saison (nur Disagreement-Spiele):")
    print(f"  {'Saison':<10} {'N':>4}  {'Modell':>7}  {'Quoten':>7}  {'Δ':>6}")
    for s in TEST_SEASONS:
        srows = [r for r in disagree if r["season"] == s]
        pm = sum(r["pts_model"] for r in srows)
        po = sum(r["pts_odds"] for r in srows)
        print(f"  {s}/{s+1}  {len(srows):4d}  {pm:7d}  {po:7d}  {pm-po:+6d}")

    # Tendenz-Wechsel bei Disagreement
    def tendency(h, a):
        return "H" if h > a else "A" if h < a else "D"

    same_tend = sum(1 for r in disagree
                    if tendency(*r["tip_model"]) == tendency(*r["tip_odds"]))
    diff_tend = len(disagree) - same_tend
    print(f"\nDisagreement-Typ:")
    print(f"  Gleiche Tendenz, anderes Ergebnis: {same_tend} ({100*same_tend/len(disagree):.1f}%)")
    print(f"  Verschiedene Tendenz:              {diff_tend} ({100*diff_tend/len(disagree):.1f}%)")

    if diff_tend > 0:
        diff_tend_rows = [r for r in disagree
                          if tendency(*r["tip_model"]) != tendency(*r["tip_odds"])]
        pm_dt = sum(r["pts_model"] for r in diff_tend_rows)
        po_dt = sum(r["pts_odds"] for r in diff_tend_rows)
        print(f"  Auf Verschieden-Tendenz-Spielen: Modell {pm_dt} vs. Quoten {po_dt} (Δ {pm_dt-po_dt:+d})")

    print(f"\nLaufzeit: {(time.time()-t_start)/60:.1f} Min")


if __name__ == "__main__":
    main()
