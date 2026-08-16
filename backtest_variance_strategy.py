#!/usr/bin/env python3
"""Standings-abhängige Varianz-Strategie — Monte-Carlo-Scaffold (Backlog #1).

Frage (EXPERIMENTS.md, Literatur-Review 2026-07-14, Punkt 5):
Ab welchem Rückstand und wie vielen Restspieltagen schlägt varianzreiches
Tippen die EV-Maximierung auf P(Runde gewinnen)? Das betrifft die Strategie
bei *fixen* Wahrscheinlichkeiten und umgeht damit das Markt-Ceiling komplett.

Ansatz
------
- DGP = reine Dixon-Coles-Score-Matrix pro Spiel (null Datenkosten, self-contained).
  Modell trainiert bis zum Split-Spieltag (kein Leakage), gilt für alle Restspiele.
- Varianz-Regler γ: pro Spiel Tipp = argmax(EV + γ·Std). γ=0 ⇒ Status quo (EV-max);
  γ>0 ⇒ höhere Punktevarianz (tendenziell exakte/Tail-Ergebnisse).
- Feld: N casual Gegner, tippen per Softmax über EV mit Temperatur T
  (T→0 scharf/EV-max, groß = zufällig). Startstände: Leader +d vor uns,
  restliches Feld gleichverteilt in [0, d].
- Monte Carlo: pro Trial Ergebnisse ~ DGP und Gegner-Tipps ~ Softmax ziehen,
  Kicktipp-Punkte vergeben, prüfen ob wir vorne stehen (Tie ⇒ Win-Share).

Ausgabe: für jedes (Rückstand d, Restspieltage R) das γ* mit maximaler
P(win) und die Verbesserung ΔP gegenüber EV-max (γ=0).

STATUS: Scaffold. Die Feld-Modellierung (Softmax-EV-Gegner) ist die zentrale
Annahme und gehört als Nächstes gegen empirische Kicktipp-Tippverteilungen
kalibriert. Ergebnisse sind vorläufig — erst nach Kalibrierung nach EXPERIMENTS.md.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt

# Kandidaten-Tipps: (th, ta) mit th,ta in 0..MAX_TIP_GOALS
K = kt.MAX_TIP_GOALS + 1
CANDIDATES = [(th, ta) for th in range(K) for ta in range(K)]  # len K*K
NR = (kt.MAX_GOALS + 1) ** 2  # Anzahl möglicher Ergebnisse (flach)

# PTS[c, r] = Kicktipp-Punkte, wenn man Kandidat c tippt und Ergebnis r eintritt.
_PT = kt._POINTS_TABLE  # [th, ta, rh, ra]
PTS = np.zeros((len(CANDIDATES), NR))
for ci, (th, ta) in enumerate(CANDIDATES):
    PTS[ci] = _PT[th, ta].reshape(NR)


def fixture_stats(prob_mat: np.ndarray):
    """Pro Spiel: flache Ergebnis-Wahrscheinlichkeit p(81,) + EV/Std je Kandidat."""
    p = prob_mat.reshape(NR)
    ev = PTS @ p                          # E[Punkte] je Kandidat
    e2 = (PTS ** 2) @ p                   # E[Punkte²]
    std = np.sqrt(np.clip(e2 - ev ** 2, 0, None))
    return p, ev, std


def focal_tip(ev, std, gamma):
    """Varianz-getilteter Tipp-Index: argmax(EV + γ·Std)."""
    return int(np.argmax(ev + gamma * std))


def opponent_probs(ev, temp):
    """Softmax-Tippverteilung eines casual Gegners über die Kandidaten."""
    z = ev / max(temp, 1e-6)
    z -= z.max()
    w = np.exp(z)
    return w / w.sum()


def build_fixtures(season, remaining, rng):
    """Trainiert ein Modell am Split und liefert (p, ev, std) je Restspiel.

    Nimmt die letzten `remaining` Spieltage der Saison als offene Runde.
    """
    all_matches = kt.load_all_matches(season)
    season_bl1 = [m for m in all_matches
                  if m["league"] == "bl1" and m["season"] == season]
    max_md = max(m["matchday"] for m in season_bl1)
    split_md = max_md - remaining + 1

    training = [m for m in all_matches
                if not (m["league"] == "bl1" and m["season"] == season
                        and m["matchday"] >= split_md)]
    if len(training) < kt.MIN_MATCHES:
        raise SystemExit(f"Zu wenig Trainingsdaten ({len(training)}).")

    ref = min(m["date"] for m in season_bl1 if m["matchday"] == split_md)
    model = kt.fit_dixon_coles(training, ref)

    fixtures = []
    for m in season_bl1:
        if m["matchday"] < split_md:
            continue
        h, a = m["home"], m["away"]
        if h not in model["attack"] or a not in model["attack"]:
            continue
        fixtures.append(fixture_stats(kt.score_matrix(h, a, model)))
    return fixtures


def simulate(fixtures, gamma, deficit, n_opp, temp, trials, rng):
    """P(focal gewinnt die Runde) für Strategie γ, Startrückstand `deficit`."""
    # Startstände der Gegner relativ zu uns: Leader exakt +deficit, Rest in [0,d].
    leads = rng.uniform(0, deficit, size=n_opp)
    if n_opp > 0:
        leads[0] = deficit
    opp_total = np.tile(leads[:, None], (1, trials)).astype(float)  # (N, trials)
    focal_total = np.zeros(trials)

    for p, ev, std in fixtures:
        # Ergebnis-Ziehung ~ DGP
        results = rng.choice(NR, size=trials, p=p)              # (trials,)
        # Unser Tipp (deterministisch aus γ) → Punkte je Trial
        c_focal = focal_tip(ev, std, gamma)
        focal_total += PTS[c_focal, results]
        # Gegner-Tipps ~ Softmax(EV/T), pro Trial neu gezogen
        if n_opp:
            q = opponent_probs(ev, temp)
            opp_tips = rng.choice(len(CANDIDATES), size=(n_opp, trials), p=q)
            opp_total += PTS[opp_tips, results]               # (N, trials)

    if n_opp == 0:
        return 1.0
    best_opp = opp_total.max(axis=0)                           # (trials,)
    wins = focal_total > best_opp
    ties = focal_total == best_opp
    n_tied_at_top = (opp_total == best_opp).sum(axis=0) + 1    # inkl. uns
    win_share = wins.astype(float) + ties / n_tied_at_top
    return float(win_share.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--opponents", type=int, default=15)
    ap.add_argument("--temp", type=float, default=0.5,
                    help="Feld-Temperatur (klein=scharf, groß=casual)")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deficits", type=int, nargs="+", default=[0, 3, 6, 10, 15])
    ap.add_argument("--remaining", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    print(f"Saison {args.season}/{args.season+1} | Feld: {args.opponents} Gegner, "
          f"T={args.temp} | {args.trials} Trials | γ-Grid {args.gammas}\n")

    for R in args.remaining:
        rng = np.random.default_rng(args.seed)
        fixtures = build_fixtures(args.season, R, rng)
        # Sanity: mittlere Eigenpunkte je γ (EV-max sollte E[Pkt] maximieren)
        print(f"═══ Restspieltage R={R}  ({len(fixtures)} Spiele) ═══")
        print(f"{'Rückstand d':>12} | " +
              " ".join(f"γ={g:<5}" for g in args.gammas) + " |  γ*   ΔP(win)")
        print("-" * (14 + 8 * len(args.gammas) + 18))
        for d in args.deficits:
            pwins = [simulate(fixtures, g, d, args.opponents, args.temp,
                              args.trials, np.random.default_rng(args.seed + d))
                     for g in args.gammas]
            p0 = pwins[0]  # Baseline γ=0 (EV-max)
            best_i = int(np.argmax(pwins))
            star = args.gammas[best_i]
            dp = pwins[best_i] - p0
            cells = " ".join(f"{p:.3f}" for p in pwins)
            flag = "" if best_i == 0 else "  ←"
            print(f"{d:>12} | {cells} |  {star:<4} {dp:+.3f}{flag}")
        print()


if __name__ == "__main__":
    main()
