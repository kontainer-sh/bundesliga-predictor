#!/usr/bin/env python3
"""Standings-abhängige Varianz-Strategie — Monte-Carlo (Backlog #1).

Frage (EXPERIMENTS.md, Literatur-Review 2026-07-14, Punkt 5):
Ab welchem Rückstand und wie vielen Restspieltagen schlägt varianzreiches Tippen
die EV-Maximierung auf P(Runde gewinnen)? Das betrifft die Strategie bei *fixen*
Wahrscheinlichkeiten und umgeht damit das Markt-Ceiling komplett.

Ansatz
------
- DGP = reine Dixon-Coles-Score-Matrix pro Spiel (null Datenkosten, self-contained).
  Modell trainiert bis zum Split-Spieltag (`training_split`), gilt für alle Restspiele.
- Varianz-Regler γ: pro Spiel Tipp = argmax(EV + γ·Std). γ=0 ⇒ Status quo (EV-max);
  γ>0 ⇒ höhere Punktevarianz (tendenziell exakte/Tail-Ergebnisse).
- **Heterogenes Feld:** N Gegner mit je eigener Temperatur T_i (Softmax über EV).
  Ein reales Kicktipp-Feld mischt scharfe und casual Tipper — nicht ein einzelnes T.
  Die T_i werden so gezogen, dass die *erwartete* Feld-Punktzahl/Spiel einen Ziel-
  bereich trifft (kalibriert an öffentlichen Ankern: naiv „2:1" ≈ 0.67, EV-max ≈ 0.81).
- Startstände: Leader +d vor uns, restliches Feld gleichverteilt in [0, d].
- Monte Carlo: Ergebnisse ~ DGP und Feld-Tipps EINMAL pro Restspieltag-Zahl R ziehen
  (dieselben Ziehungen für alle γ/d → varianzarme, gepaarte Vergleiche), dann
  P(focal gewinnt) je (γ, d) auswerten. Tie ⇒ Win-Share.

`--calibrate` druckt die Abbildung T → erwartete Punkte/Spiel (zum Anker-Setzen).

STATUS: Feld-Kalibrierung an *Punkte*-Anker (öffentlich). Eine Kalibrierung an
echte Kicktipp-*Tippverteilungen* würde ligeninterne Daten brauchen (bleiben lokal).
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
import kicktipp as kt

K = kt.MAX_TIP_GOALS + 1
CANDS = [(th, ta) for th in range(K) for ta in range(K)]
NR = (kt.MAX_GOALS + 1) ** 2

PTS = np.zeros((len(CANDS), NR))
for ci, (th, ta) in enumerate(CANDS):
    PTS[ci] = kt._POINTS_TABLE[th, ta].reshape(NR)

FIELDS = {  # Temperatur-Bereiche → gezogen U(lo, hi) je Gegner
    "sharp":  (0.05, 0.20),   # fast alle nahe EV-max
    "mixed":  (0.10, 0.80),   # ein paar scharfe, viele mittlere
    "casual": (0.30, 1.50),   # überwiegend verrauscht
}


def fixture_stats(prob_mat):
    p = prob_mat.reshape(NR)
    ev = PTS @ p
    std = np.sqrt(np.clip((PTS ** 2) @ p - ev ** 2, 0, None))
    return p, ev, std


def focal_tip(ev, std, gamma):
    return int(np.argmax(ev + gamma * std))


def opponent_probs(ev, temp):
    z = ev / max(temp, 1e-6)
    z -= z.max()
    w = np.exp(z)
    return w / w.sum()


def build_fixtures(season, remaining):
    allm = kt.load_all_matches(season)
    sb = [m for m in allm if m["league"] == "bl1" and m["season"] == season]
    split_md = max(m["matchday"] for m in sb) - remaining + 1
    ref = min(m["date"] for m in sb if m["matchday"] == split_md)
    model = kt.fit_dixon_coles(kt.training_split(allm, season, split_md, ref_date=ref), ref)
    fx = []
    for m in sb:
        if m["matchday"] < split_md:
            continue
        h, a = m["home"], m["away"]
        if h in model["attack"] and a in model["attack"]:
            fx.append(fixture_stats(kt.score_matrix(h, a, model)))
    return fx


def expected_ppg(fixtures, temp):
    """Erwartete Punkte/Spiel eines Softmax-T-Gegners (analytisch)."""
    tot = sum(float(opponent_probs(ev, temp) @ ev) for _, ev, _ in fixtures)
    return tot / len(fixtures)


def evmax_ppg(fixtures):
    return sum(float(ev.max()) for _, ev, _ in fixtures) / len(fixtures)


def sample_field(fixtures, opp_temps, trials, rng):
    """Zieht Ergebnisse + Feld-Tipps EINMAL. Gibt (results je Spiel,
    focal-Punkte je Kandidat-pro-Spiel, Gegner-Zusatzpunkte N×trials)."""
    n_opp = len(opp_temps)
    results = [rng.choice(NR, size=trials, p=p) for p, _, _ in fixtures]
    opp_total = np.zeros((n_opp, trials))
    for g_i, (_, ev, _) in enumerate(fixtures):
        res = results[g_i]
        for i, Ti in enumerate(opp_temps):
            tips = rng.choice(len(CANDS), size=trials, p=opponent_probs(ev, Ti))
            opp_total[i] += PTS[tips, res]
    return results, opp_total


def focal_added(fixtures, results, gamma):
    """Focal-Zusatzpunkte je Trial für Strategie γ (deterministische Tipps)."""
    out = np.zeros(len(results[0]))
    for g_i, (_, ev, std) in enumerate(fixtures):
        out += PTS[focal_tip(ev, std, gamma), results[g_i]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--opponents", type=int, default=19)  # 20er-Runde inkl. uns
    ap.add_argument("--field", choices=list(FIELDS), default="casual")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deficits", type=int, nargs="+", default=[0, 3, 6, 10, 15])
    ap.add_argument("--remaining", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--calibrate", action="store_true",
                    help="nur T→Punkte/Spiel drucken (Feld-Anker setzen)")
    args = ap.parse_args()

    if args.calibrate:
        fx = build_fixtures(args.season, max(args.remaining))
        print(f"\nKalibrierung Saison {args.season}/{args.season+1} "
              f"({len(fx)} Spiele):\n")
        print(f"  EV-max (T→0):        {evmax_ppg(fx):.3f} Pkt/Spiel")
        for T in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0):
            print(f"  T={T:<4}:              {expected_ppg(fx, T):.3f} Pkt/Spiel")
        for name, (lo, hi) in FIELDS.items():
            mid = (lo + hi) / 2
            print(f"  Feld '{name}' (T∈[{lo},{hi}], Mitte {mid}): "
                  f"{expected_ppg(fx, mid):.3f} Pkt/Spiel")
        return

    lo, hi = FIELDS[args.field]
    opp_temps = np.random.default_rng(args.seed).uniform(lo, hi, size=args.opponents)

    print(f"Saison {args.season}/{args.season+1} | Feld '{args.field}' "
          f"({args.opponents} Gegner, T∈[{lo},{hi}]) | {args.trials} Trials\n")

    for R in args.remaining:
        fx = build_fixtures(args.season, R)
        field_ppg = np.mean([expected_ppg(fx, T) for T in opp_temps])
        # Ergebnisse + Feld-Tipps EINMAL ziehen; für alle γ und d wiederverwendet.
        results, opp_total = sample_field(fx, opp_temps, args.trials,
                                          np.random.default_rng(args.seed + R))
        foc = {g: focal_added(fx, results, g) for g in args.gammas}

        print(f"═══ R={R} Restspieltage ({len(fx)} Spiele) | "
              f"Feld Ø {field_ppg:.3f} vs EV-max {evmax_ppg(fx):.3f} Pkt/Spiel ═══")
        print(f"{'Rückstand d':>12} | " + " ".join(f"γ={g:<4}" for g in args.gammas)
              + " |  γ*   ΔP")
        print("-" * (14 + 8 * len(args.gammas) + 16))
        for d in args.deficits:
            # Startstände einmal pro d ziehen → γ-Vergleich sauber gepaart.
            leads = np.random.default_rng(args.seed + 7 * d + R).uniform(0, d, size=args.opponents)
            if args.opponents:
                leads[0] = d
            opp = opp_total + leads[:, None]
            best = opp.max(axis=0)
            ties = (opp == best).sum(axis=0) + 1
            pw = [float((foc[g] > best).mean() + ((foc[g] == best) / ties).mean())
                  for g in args.gammas]
            i = int(np.argmax(pw))
            flag = "" if i == 0 else "  ←"
            print(f"{d:>12} | " + " ".join(f"{p:.3f}" for p in pw)
                  + f" |  {args.gammas[i]:<4} {pw[i]-pw[0]:+.3f}{flag}")
        print()


if __name__ == "__main__":
    main()
