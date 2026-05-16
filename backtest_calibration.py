#!/usr/bin/env python3
"""Calibration-Test der Score-Matrix.

Vergleicht drei Strategien probabilistisch (nicht argmax):
- λ=0.0  reines DC
- λ=0.7  Produktion (70% Quoten + 30% DC)
- λ=1.0  reine Quoten

Aggregiert die 8×8-Score-Matrix zu Markt-Wahrscheinlichkeiten (1X2, Over 2.5, BTTS)
und vergleicht gegen realisierte Ergebnisse. Metriken: Brier, LogLoss, ECE,
plus Reliability-Tabelle.

Sagt diagnostisch *warum* eine Strategie funktioniert oder nicht — z.B. ob die
Matrix systematisch über-/unterzuversichtlich ist und ob Recalibration sinnvoll
wäre.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt


TEST_SEASONS = [2022, 2023, 2024]
STRATEGIES = [
    ("DC pur (λ=0.0)", 0.0),
    ("Mix Prod (λ=0.7)", 0.7),
    ("Quoten pur (λ=1.0)", 1.0),
]
RELIABILITY_BINS = np.linspace(0.0, 1.0, 11)


def mix_matrix(dc_mat: np.ndarray, o_mat: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        m = dc_mat
    elif lam == 1.0:
        m = o_mat
    else:
        m = (1 - lam) * dc_mat + lam * o_mat
    return m / m.sum()


def market_probs(mat: np.ndarray) -> dict:
    """1X2, Over 2.5, BTTS aus der Score-Matrix."""
    n = mat.shape[0]
    h_idx, a_idx = np.indices((n, n))
    return {
        "home":  float(mat[h_idx > a_idx].sum()),
        "draw":  float(mat[h_idx == a_idx].sum()),
        "away":  float(mat[h_idx < a_idx].sum()),
        "over25":  float(mat[(h_idx + a_idx) > 2].sum()),
        "btts": float(mat[(h_idx >= 1) & (a_idx >= 1)].sum()),
    }


def actual_outcomes(h: int, a: int) -> dict:
    return {
        "home": int(h > a),
        "draw": int(h == a),
        "away": int(h < a),
        "over25": int((h + a) > 2),
        "btts": int(h >= 1 and a >= 1),
    }


def brier_multiclass(probs_3: np.ndarray, outcomes_3: np.ndarray) -> float:
    """Brier-Score für 1X2 (3-Klassen). Probs Nx3, Outcomes Nx3 (one-hot)."""
    return float(((probs_3 - outcomes_3) ** 2).sum(axis=1).mean())


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def logloss_multiclass(probs_3: np.ndarray, outcomes_3: np.ndarray, eps=1e-12) -> float:
    p = np.clip(probs_3, eps, 1 - eps)
    return float(-(outcomes_3 * np.log(p)).sum(axis=1).mean())


def logloss_binary(p: np.ndarray, y: np.ndarray, eps=1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(p: np.ndarray, y: np.ndarray, bins=RELIABILITY_BINS) -> float:
    """Expected Calibration Error (binary): gewichtete |conf - acc|."""
    ece_val = 0.0
    n = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece_val += (mask.sum() / n) * abs(conf - acc)
    return float(ece_val)


def reliability_table(p: np.ndarray, y: np.ndarray, bins=RELIABILITY_BINS):
    """Pro Bin: N, mean predicted prob, empirical freq."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            out.append((lo, hi, 0, None, None))
        else:
            out.append((lo, hi, int(mask.sum()), float(p[mask].mean()), float(y[mask].mean())))
    return out


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

            real = actual_outcomes(m["home_goals"], m["away_goals"])
            row = {"real": real}
            for _, lam in STRATEGIES:
                mat = mix_matrix(dc_mat, o_mat, lam)
                row[lam] = market_probs(mat)
            rows.append(row)
    return rows


def main():
    t_start = time.time()
    all_rows = []
    for s in TEST_SEASONS:
        all_rows.extend(run_season(s))

    n = len(all_rows)
    print(f"\n{'='*82}")
    print(f"CALIBRATION-TEST — Score-Matrix vs. realisierte Outcomes")
    print(f"Aggregat über Saisons {TEST_SEASONS}, {n} Spiele\n{'='*82}\n")

    # ── 1X2 (multi-class) ────────────────────────────────────────────────────
    print("1X2-Markt (Multi-Class):")
    print(f"{'Strategie':<22} {'Brier':>8} {'LogLoss':>9} {'ECE-H':>7} {'ECE-D':>7} {'ECE-A':>7}")
    print("─" * 70)
    for name, lam in STRATEGIES:
        probs_3 = np.array([[r[lam]["home"], r[lam]["draw"], r[lam]["away"]] for r in all_rows])
        out_3 = np.array([[r["real"]["home"], r["real"]["draw"], r["real"]["away"]] for r in all_rows])
        br = brier_multiclass(probs_3, out_3)
        ll = logloss_multiclass(probs_3, out_3)
        ece_h = ece(probs_3[:, 0], out_3[:, 0])
        ece_d = ece(probs_3[:, 1], out_3[:, 1])
        ece_a = ece(probs_3[:, 2], out_3[:, 2])
        print(f"{name:<22} {br:>8.4f} {ll:>9.4f} {ece_h:>7.4f} {ece_d:>7.4f} {ece_a:>7.4f}")

    # ── Over 2.5 / BTTS (binary) ─────────────────────────────────────────────
    print("\nBinäre Märkte:")
    print(f"{'Strategie':<22} {'Markt':<8} {'Brier':>8} {'LogLoss':>9} {'ECE':>7} "
          f"{'mean(p)':>9} {'mean(y)':>9}")
    print("─" * 82)
    for name, lam in STRATEGIES:
        for market in ["over25", "btts"]:
            p = np.array([r[lam][market] for r in all_rows])
            y = np.array([r["real"][market] for r in all_rows]).astype(float)
            print(f"{name:<22} {market:<8} {brier_binary(p, y):>8.4f} "
                  f"{logloss_binary(p, y):>9.4f} {ece(p, y):>7.4f} "
                  f"{p.mean():>9.4f} {y.mean():>9.4f}")

    # ── Reliability-Tabelle für Home-Win bei Produktion ──────────────────────
    print("\nReliability-Tabelle: P(Home Win) bei Produktion λ=0.7")
    print(f"{'Bin':<14} {'N':>5} {'⟨p⟩':>8} {'⟨y⟩':>8} {'Δ':>8}")
    print("─" * 50)
    p_home = np.array([r[0.7]["home"] for r in all_rows])
    y_home = np.array([r["real"]["home"] for r in all_rows]).astype(float)
    for lo, hi, n_bin, pm, ym in reliability_table(p_home, y_home):
        if n_bin == 0:
            print(f"[{lo:.1f}, {hi:.1f}){'':<6} {n_bin:>5}")
        else:
            print(f"[{lo:.1f}, {hi:.1f}){'':<6} {n_bin:>5} {pm:>8.4f} {ym:>8.4f} {ym-pm:>+8.4f}")

    print(f"\nLaufzeit: {(time.time()-t_start)/60:.1f} Min")


if __name__ == "__main__":
    main()
