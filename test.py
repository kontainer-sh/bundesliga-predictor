#!/usr/bin/env python3
"""Smoke-Tests für die Kernlogik."""
import sys
sys.path.insert(0, ".")
import numpy as np
import kicktipp as kt

errors = 0

def check(name, condition):
    global errors
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}")
        errors += 1

# --- Punkteregeln ---
print("Punkteregeln:")
check("Exakt 2:1 → 3 Pkt", kt.kicktipp_points(2, 1, 2, 1) == kt.POINTS_EXACT)
check("Exakt 0:0 → 3 Pkt", kt.kicktipp_points(0, 0, 0, 0) == kt.POINTS_EXACT)
check("Differenz 2:1 vs 3:2 → 2 Pkt", kt.kicktipp_points(2, 1, 3, 2) == kt.POINTS_GOAL_DIFF)
check("Differenz 0:1 vs 0:1 → 3 Pkt (exakt!)", kt.kicktipp_points(0, 1, 0, 1) == kt.POINTS_EXACT)
check("Remis 1:1 vs 0:0 → 2 Pkt", kt.kicktipp_points(1, 1, 0, 0) == kt.POINTS_DRAW_TENDENCY)
check("Remis 2:2 vs 1:1 → 2 Pkt", kt.kicktipp_points(2, 2, 1, 1) == kt.POINTS_DRAW_TENDENCY)
check("Tendenz 1:0 vs 3:1 → 1 Pkt", kt.kicktipp_points(1, 0, 3, 1) == kt.POINTS_TENDENCY)
check("Tendenz 0:1 vs 1:3 → 1 Pkt", kt.kicktipp_points(0, 1, 1, 3) == kt.POINTS_TENDENCY)
check("Falsch 1:0 vs 0:1 → 0 Pkt", kt.kicktipp_points(1, 0, 0, 1) == 0)
check("Falsch 2:1 vs 1:1 → 0 Pkt", kt.kicktipp_points(2, 1, 1, 1) == 0)
check("Falsch 0:0 vs 1:0 → 0 Pkt", kt.kicktipp_points(0, 0, 1, 0) == 0)
print()

# --- Score-Matrix Orientierung ---
print("Score-Matrix (Home/Away korrekt):")
# Starker Heimfavorit → Matrix muss Heimsieg bevorzugen
model = {
    "attack": {"Home": 0.5, "Away": -0.3},
    "defense": {"Home": -0.2, "Away": 0.3},
    "home_adv": 0.3,
    "rho": -0.1,
}
mat = kt.score_matrix("Home", "Away", model)
p_home = np.sum(mat[np.tril_indices(kt.MAX_GOALS + 1, k=-1)])
p_draw = np.trace(mat)
p_away = np.sum(mat[np.triu_indices(kt.MAX_GOALS + 1, k=1)])
check(f"Heimfavorit: P(H)={p_home:.2f} > P(A)={p_away:.2f}", p_home > p_away)
check(f"Summe = 1.0", abs(mat.sum() - 1.0) < 1e-6)
print()

# --- Odds-Score-Matrix Orientierung ---
print("Odds-Score-Matrix (Home/Away korrekt):")
# Klarer Auswärtssieg in Quoten → Matrix muss Auswärts bevorzugen
mat_odds = kt.odds_to_score_matrix(0.2, 0.2, 0.6)
p_home_o = np.sum(mat_odds[np.tril_indices(kt.MAX_GOALS + 1, k=-1)])
p_away_o = np.sum(mat_odds[np.triu_indices(kt.MAX_GOALS + 1, k=1)])
check(f"Auswärtsfavorit: P(A)={p_away_o:.2f} > P(H)={p_home_o:.2f}", p_away_o > p_home_o)

mat_odds2 = kt.odds_to_score_matrix(0.6, 0.2, 0.2)
p_home_o2 = np.sum(mat_odds2[np.tril_indices(kt.MAX_GOALS + 1, k=-1)])
p_away_o2 = np.sum(mat_odds2[np.triu_indices(kt.MAX_GOALS + 1, k=1)])
check(f"Heimfavorit: P(H)={p_home_o2:.2f} > P(A)={p_away_o2:.2f}", p_home_o2 > p_away_o2)
check(f"Summe = 1.0", abs(mat_odds.sum() - 1.0) < 1e-6)
print()

# --- _find_odds mit umgekehrter Paarung ---
print("Odds-Matching:")
odds = {("Team A", "Team B"): {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}}
check("Direkt gefunden", kt._find_odds(odds, "Team A", "Team B") is not None)
reversed_od = kt._find_odds(odds, "Team B", "Team A")
check("Umgekehrt gefunden", reversed_od is not None)
check("Umgekehrt: p_home/p_away getauscht",
      reversed_od["p_home"] == 0.2 and reversed_od["p_away"] == 0.5)
check("Nicht gefunden", kt._find_odds(odds, "Team A", "Team C") is None)
check("Leeres Dict", kt._find_odds({}, "A", "B") is None)
check("None", kt._find_odds(None, "A", "B") is None)
print()

# --- best_tip Plausibilität ---
print("Tipp-Optimierung:")
# Bei symmetrischen Teams sollte Unentschieden rauskommen
model_sym = {
    "attack": {"A": 0.0, "B": 0.0},
    "defense": {"A": 0.0, "B": 0.0},
    "home_adv": 0.0,
    "rho": -0.1,
}
th, ta, ev = kt.best_tip("A", "B", model_sym)
check(f"Symmetrisch → Remis-Tipp ({th}:{ta})", th == ta)

# Bei starkem Heimfavorit sollte Heimsieg rauskommen
th2, ta2, ev2 = kt.best_tip("Home", "Away", model)
check(f"Heimfavorit → Heimsieg-Tipp ({th2}:{ta2})", th2 > ta2)
print()

# --- Team-Name-Mapping ---
print("Team-Name-Mapping:")
check("Bayern", kt._normalize_team("Bayern Munich") == "FC Bayern München")
check("Gladbach", kt._normalize_team("M'gladbach") == "Borussia Mönchengladbach")
check("Mainz", kt._normalize_team("FSV Mainz 05") == "1. FSV Mainz 05")
check("Leverkusen", kt._normalize_team("Bayer Leverkusen") == "Bayer 04 Leverkusen")
check("Heidenheim", kt._normalize_team("1. FC Heidenheim") == "1. FC Heidenheim 1846")
check("Unbekannt bleibt", kt._normalize_team("Unbekannt FC") == "Unbekannt FC")
# Aufsteiger 2026/27 — Odds-API-Kurznamen (Regression: fielen sonst auf "Nur Modell")
check("Elversberg", kt._normalize_team("Elversberg") == "SV 07 Elversberg")
check("SC Paderborn", kt._normalize_team("SC Paderborn") == "SC Paderborn 07")
print()

# --- Trainings-Split (Data-Leakage + Regression gegen den date.year-Bug) ---
print("Trainings-Split:")
from datetime import datetime

def _mk(mid, season, league, md, year, month):
    return {"id": mid, "matchday": md, "date": datetime(year, month, 1),
            "home": "H", "away": "A", "home_goals": 1, "away_goals": 0,
            "league": league, "season": season}

# Laufende Saison = 2026/27. Die Vorsaison-Rückrunde (2025/26) wird im
# Kalenderjahr 2026 gespielt — genau daran ist der alte date.year-Filter zerbrochen.
_ms = [
    _mk("vor-rueck", 2025, "bl1", 30, 2026, 3),   # Vorsaison-Rückrunde, Jahr 2026
    _mk("vor-hin",   2025, "bl1", 5,  2025, 9),    # Vorsaison-Hinrunde, Jahr 2025
    _mk("cur-md1",   2026, "bl1", 1,  2026, 8),    # laufend, Spieltag 1
    _mk("cur-md3",   2026, "bl1", 3,  2026, 9),    # laufend, Spieltag 3
    _mk("cur-md7",   2026, "bl1", 7,  2026, 10),   # laufend, Spieltag 7
    _mk("cur-bl2",   2026, "bl2", 8,  2026, 8),    # laufend, 2. Liga (andere Liga)
]

def _ids(matches):
    return {m["id"] for m in matches}

# Spieltag 1: laufende 1.-Liga-Saison komplett raus, alles andere bleibt
s1 = _ids(kt.training_split(_ms, 2026, 1))
check("MD1: Vorsaison-Rückrunde bleibt (der Bug!)", "vor-rueck" in s1)
check("MD1: Vorsaison-Hinrunde bleibt", "vor-hin" in s1)
check("MD1: 2. Liga (andere Liga) bleibt", "cur-bl2" in s1)
check("MD1: laufende Saison MD1 raus", "cur-md1" not in s1)
check("MD1: laufende Saison MD7 raus", "cur-md7" not in s1)

# Spieltag 5: nur laufende 1.-Liga-Spieltage >= 5 raus, MD1/MD3 bleiben
s5 = _ids(kt.training_split(_ms, 2026, 5))
check("MD5: laufende MD3 bleibt (3 < 5)", "cur-md3" in s5)
check("MD5: laufende MD7 raus (7 >= 5)", "cur-md7" not in s5)
check("MD5: Vorsaison-Rückrunde bleibt", "vor-rueck" in s5)

# Regression: der alte date.year-Filter hätte die Vorsaison-Rückrunde verworfen
_buggy = _ids([m for m in _ms
               if not (m["matchday"] >= 1 and m["date"].year >= 2026)])
check("Alter Bug hätte Vorsaison-Rückrunde verworfen", "vor-rueck" not in _buggy)
check("Fix behält 324-Äquivalent (mehr als der Bug)", len(s1) > len(_buggy))
print()

# --- Ergebnis ---
if errors == 0:
    print("Alle Tests bestanden.")
else:
    print(f"{errors} Test(s) fehlgeschlagen!")
    sys.exit(1)
