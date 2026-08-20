#!/usr/bin/env python3
"""Diagnose: Trägt Pinnacle-Closing (Asian Handicap + Over/Under) Information
über die aus 1X2 rekonstruierte Poisson-Score-Matrix hinaus?

Hintergrund: `odds_to_score_matrix` rekonstruiert die Score-Verteilung aus den
1X2-Closing-Quoten (2 Freiheitsgrade → λ_home, λ_away, unabhängiges Poisson).
Die Hypothese des Exact-Score-Proxys: AH-Linien fixieren die *Tordifferenz*-
Verteilung schärfer als 1X2 allein, Totals die *Gesamttor*-Verteilung — also
Information, die die 2-Parameter-Poisson-Matrix noch nicht hat. Falls die
1X2-Matrix die Markt-AH/OU-Preise schon reproduziert (und auf den realen
Ausgängen gleich gut kalibriert ist), gibt es kein Headroom → Null.

Entscheidender Test: **Brier(Markt) vs. Brier(1X2-Modell)** auf den realen
AH-/OU-Ausgängen. Schlägt der Markt das 1X2-Modell dort signifikant, existiert
verwertbare Zusatzinfo für den Proxy; sonst ist das Ceiling schon ausgeschöpft.

Rein diagnostisch, kein Prod-Change. Läuft offline aus dem CSV-Cache
(football-data.co.uk-Zeilen enthalten Ergebnis + AH/OU/1X2 zusammen, kein
Team-Mapping nötig).

Aufruf:  python backtest_score_markets.py [season_from] [season_to]
Default: 2019 2025
"""
import csv
import sys

import numpy as np

import kicktipp as kt


def devig2(o_a: float, o_b: float) -> float:
    """Vig-freie Wahrscheinlichkeit der Seite A eines 2-Wege-Marktes."""
    ia, ib = 1.0 / o_a, 1.0 / o_b
    return ia / (ia + ib)


def margin_pmf(mat: np.ndarray) -> dict:
    """P(Tordifferenz = m) aus Score-Matrix (m = Heim - Auswärts)."""
    n = mat.shape[0]
    pmf = {}
    for gh in range(n):
        for ga in range(n):
            m = gh - ga
            pmf[m] = pmf.get(m, 0.0) + mat[gh, ga]
    return pmf


def over_prob(mat: np.ndarray, line: float) -> float:
    n = mat.shape[0]
    p = 0.0
    for gh in range(n):
        for ga in range(n):
            if gh + ga > line:
                p += mat[gh, ga]
    return p


def ah_model_prob(pmf: dict, h: float):
    """Modell-P(Heim deckt AH-Linie h | entscheidend), plus Push-Masse.

    Konvention (validiert im Lauf über Kalibrierung): positives AHCh = Heim
    erhält Vorsprung h; Heim deckt, wenn (m + h) > 0, Push wenn (m + h) == 0.
    Nur Halb-/Ganzlinien (2*h ganzzahlig); Viertellinien werden vom Aufrufer
    ausgeschlossen.
    """
    p_win = sum(p for m, p in pmf.items() if m + h > 1e-9)
    p_push = sum(p for m, p in pmf.items() if abs(m + h) < 1e-9)
    p_lose = sum(p for m, p in pmf.items() if m + h < -1e-9)
    denom = p_win + p_lose
    return (p_win / denom if denom > 0 else 0.5), p_push


def load_rows(season: int) -> list[dict]:
    """Rohzeilen mit vollständigem Pinnacle-Closing (1X2 + OU2.5 + AH)."""
    path = kt.CACHE_DIR / f"odds_D1_{season}.csv"
    if not path.exists():
        kt.fetch_odds_csv(season)  # legt Cache an
    content = path.read_text(encoding="utf-8-sig")
    out = []
    for row in csv.DictReader(content.splitlines()):
        def f(key):
            try:
                return float(row.get(key) or 0)
            except (ValueError, TypeError):
                return 0.0
        psh, psd, psa = f("PSCH"), f("PSCD"), f("PSCA")
        pov, pun = f("PC>2.5"), f("PC<2.5")
        ahh, aha = f("PCAHH"), f("PCAHA")
        try:
            ah_line = float(row.get("AHCh"))
        except (ValueError, TypeError):
            ah_line = None
        if min(psh, psd, psa) <= 0:
            continue
        try:
            hg, ag = int(row.get("FTHG")), int(row.get("FTAG"))
        except (ValueError, TypeError):
            continue
        inv = 1/psh + 1/psd + 1/psa
        out.append({
            "p_home": (1/psh)/inv, "p_draw": (1/psd)/inv, "p_away": (1/psa)/inv,
            "hg": hg, "ag": ag,
            "p_over_mkt": devig2(pov, pun) if pov > 0 and pun > 0 else None,
            "ah_line": ah_line,
            "p_ah_home_mkt": devig2(ahh, aha) if ahh > 0 and aha > 0 else None,
        })
    return out


def brier(p, y):
    return np.mean((np.asarray(p) - np.asarray(y)) ** 2)


def main():
    s_from = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    s_to = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    rows = []
    for s in range(s_from, s_to + 1):
        try:
            rows += load_rows(s)
        except Exception as e:
            print(f"  Warnung: Saison {s} nicht geladen ({e})")
    print(f"\n=== SCORE-MARKETS-DIAGNOSE Saisons {s_from}/{s_from+1}–{s_to}/{s_to+1} ===")
    print(f"  {len(rows)} Spiele mit 1X2-Closing\n")

    # Sammelbehälter
    ou_mod, ou_mkt, ou_real = [], [], []
    ah_mod, ah_mkt, ah_real = [], [], []
    ah_used = ah_quarter = ah_push = 0

    for r in rows:
        mat = kt.odds_to_score_matrix(r["p_home"], r["p_draw"], r["p_away"])  # nur 1X2
        pmf = margin_pmf(mat)

        # --- Over/Under 2.5 (Halblinie, kein Push) ---
        if r["p_over_mkt"] is not None:
            ou_mod.append(over_prob(mat, 2.5))
            ou_mkt.append(r["p_over_mkt"])
            ou_real.append(1.0 if (r["hg"] + r["ag"]) > 2.5 else 0.0)

        # --- Asian Handicap (nur Halb-/Ganzlinien) ---
        h = r["ah_line"]
        if r["p_ah_home_mkt"] is not None and h is not None:
            if abs((2 * h) - round(2 * h)) > 1e-9:  # Viertellinie → auslassen
                ah_quarter += 1
            else:
                p_mod, p_push = ah_model_prob(pmf, h)
                m = r["hg"] - r["ag"]
                if abs(m + h) < 1e-9:      # realer Push → aus Kalibrierung raus
                    ah_push += 1
                else:
                    ah_used += 1
                    ah_mod.append(p_mod)
                    ah_mkt.append(r["p_ah_home_mkt"])
                    ah_real.append(1.0 if (m + h) > 0 else 0.0)

    def block(name, mod, mkt, real):
        mod, mkt, real = map(np.asarray, (mod, mkt, real))
        n = len(mod)
        print(f"— {name}  (n={n}) —")
        print(f"    Modell vs. Markt:  MAE {np.mean(np.abs(mod-mkt)):.4f} · "
              f"mean(Modell−Markt) {np.mean(mod-mkt):+.4f} · corr {np.corrcoef(mod,mkt)[0,1]:.3f}")
        b_mod, b_mkt = brier(mod, real), brier(mkt, real)
        print(f"    Brier vs. Realität: Modell {b_mod:.4f} · Markt {b_mkt:.4f} · "
              f"Δ {b_mod-b_mkt:+.4f}  ({'Markt besser' if b_mkt<b_mod else 'Modell besser/gleich'})")
        # Paired Bootstrap auf der Brier-Differenz je Spiel
        d = (mod-real)**2 - (mkt-real)**2
        rng = np.random.default_rng(0)
        boot = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = "signifikant" if (lo > 0 or hi < 0) else "n.s."
        print(f"    Brier-Δ/Spiel {np.mean(d):+.5f}, 95%-CI [{lo:+.5f}, {hi:+.5f}]  → {sig}\n")

    block("Over/Under 2.5", ou_mod, ou_mkt, ou_real)
    print(f"  AH: {ah_used} Halb-/Ganzlinien genutzt, {ah_quarter} Viertellinien ausgelassen, "
          f"{ah_push} reale Pushes ausgeschlossen")
    block("Asian Handicap", ah_mod, ah_mkt, ah_real)

    print("Lesart: Schlägt der Markt das 1X2-Modell auf der Brier-Differenz")
    print("signifikant, trägt AH/OU-Closing echte Zusatzinfo → Headroom für den")
    print("Exact-Score-Proxy. Sonst ist die 1X2-Poisson-Matrix bereits am Ceiling.")


if __name__ == "__main__":
    main()
