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
# Kein Reverse-Fixture-Fallback (Finding 2): das Rückspiel ist eine andere Partie —
# stille Substitution (evtl. mit Zukunftsdaten) wäre gefährlicher als „keine Odds".
check("Umgekehrt → None (keine stille Substitution)",
      kt._find_odds(odds, "Team B", "Team A") is None)
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
check("SV Elversberg (Kicktipp/Odds-Variante)",
      kt._normalize_team("SV Elversberg") == "SV 07 Elversberg")
check("SC Paderborn", kt._normalize_team("SC Paderborn") == "SC Paderborn 07")
check("Schalke", kt._normalize_team("Schalke") == "FC Schalke 04")

# Aktueller BL1-Kader 2026/27 (OpenLigaDB-Namen) — muss identity-safe sein, damit
# der Odds-Join greift, wenn die Quelle den kanonischen Namen nutzt. Bei jedem
# Aufsteiger-Wechsel pflegen: erzwingt den saisonalen Mapping-Check (Betrieb).
BL1_2026_27 = [
    "FC Bayern München", "Borussia Dortmund", "Bayer 04 Leverkusen", "RB Leipzig",
    "Eintracht Frankfurt", "SC Freiburg", "TSG Hoffenheim", "1. FC Union Berlin",
    "VfB Stuttgart", "SV Werder Bremen", "1. FSV Mainz 05", "FC Augsburg",
    "Borussia Mönchengladbach", "1. FC Köln", "Hamburger SV", "FC Schalke 04",
    "SC Paderborn 07", "SV 07 Elversberg",
]
for _t in BL1_2026_27:
    check(f"identity-safe: {_t}", kt._normalize_team(_t) == _t)
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
    _mk("cur-bl2-past",   2026, "bl2", 2,  2026, 8),   # laufende 2. Liga, Aug (Vergangenheit)
    _mk("cur-bl2-future", 2026, "bl2", 20, 2026, 12),  # laufende 2. Liga, Dez (ZUKUNFT → Leak)
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

# Datums-Cutoff (ref_date): schließt den BL2-Zukunfts-Leak (Review-Finding 2026-08-21)
_cutoff = datetime(2026, 10, 1)
s_cut = _ids(kt.training_split(_ms, 2026, 1, ref_date=_cutoff))
check("Cutoff: künftiges BL2-Spiel (Dez) RAUS — der Leak", "cur-bl2-future" not in s_cut)
check("Cutoff: vergangenes BL2-Spiel (Aug) bleibt", "cur-bl2-past" in s_cut)
check("Cutoff: Vorsaison bleibt (kein date.year-Bug)",
      "vor-hin" in s_cut and "vor-rueck" in s_cut)
check("Ohne ref_date: künftiges BL2 bleibt drin (rückwärtskompatibel)",
      "cur-bl2-future" in _ids(kt.training_split(_ms, 2026, 1)))
print()

# --- _odds_val Fallback-Kette (Closing → Pre-Closing → generisch) ---
print("_odds_val (Regression gegen die O/U-Closing-Bug-Klasse):")
check("Closing bevorzugt", kt._odds_val({"PSCH": "2.0", "PSH": "1.9"}, "PSCH", "PSH", "PH") == 2.0)
check("Fallback wenn Closing fehlt", kt._odds_val({"PSH": "1.9"}, "PSCH", "PSH", "PH") == 1.9)
check("Leerer Wert übersprungen", kt._odds_val({"PSCH": "", "PSH": "1.9"}, "PSCH", "PSH") == 1.9)
check("Null übersprungen (v>0)", kt._odds_val({"PSCH": "0", "PSH": "1.9"}, "PSCH", "PSH") == 1.9)
check("Unparsebar übersprungen", kt._odds_val({"PSCH": "x", "PSH": "1.9"}, "PSCH", "PSH") == 1.9)
check("Nichts vorhanden → 0.0", kt._odds_val({}, "PSCH", "PSH") == 0.0)
print()

# --- odds_to_score_matrix reproduziert die Quoten-Inputs (KL-Fidelity, nicht nur Orientierung) ---
print("odds_to_score_matrix (Round-Trip: Matrix reproduziert die Inputs):")
def _hda(mat):
    n = kt.MAX_GOALS + 1
    return (float(np.sum(mat[np.tril_indices(n, -1)])), float(np.trace(mat)),
            float(np.sum(mat[np.triu_indices(n, 1)])))
for _p in [(0.5, 0.3, 0.2), (0.25, 0.28, 0.47), (0.7, 0.18, 0.12), (0.45, 0.27, 0.28)]:
    _m = kt.odds_to_score_matrix(*_p)
    _ph, _pd, _pa = _hda(_m)
    _err = max(abs(_ph-_p[0]), abs(_pd-_p[1]), abs(_pa-_p[2]))
    check(f"H/D/A {_p} reproduziert (max-Fehler {_err:.3f})", _err < 0.04)
# O/U-Branch: reproduziert zusätzlich P(over 2.5), Favorit bleibt Favorit
_mou = kt.odds_to_score_matrix(0.4, 0.3, 0.3, p_over=0.60, ou_line=2.5)
_n = kt.MAX_GOALS + 1
_over = float(sum(_mou[i, j] for i in range(_n) for j in range(_n) if i + j > 2.5))
check(f"O/U-Branch reproduziert P(over)=0.60 (ist {_over:.3f})", abs(_over-0.60) < 0.04)
_oh, _od, _oa = _hda(_mou)
check("O/U-Branch: Heim bleibt Favorit", _oh > _od and _oh > _oa)
print()

# --- compute_tip (der ECHTE Prod-Tipp: Modell + Odds-Mix; best_tip ist nur Modell-only) ---
print("compute_tip (Modell + Odds-Blend):")
_th, _ta, _ev = kt.compute_tip("Home", "Away", model)  # 'model' = Heimfavorit (oben def.)
check(f"Ohne Odds, Heimfav → Heimsieg-Tipp ({_th}:{_ta})", _th > _ta)
check("Tipp im gültigen Bereich [0..MAX_TIP_GOALS]",
      0 <= _th <= kt.MAX_TIP_GOALS and 0 <= _ta <= kt.MAX_TIP_GOALS)
# Symmetrisches Modell, aber Odds klar auswärts → 70%-Blend zieht auf Auswärtssieg
_odds_away = {("A", "B"): {"p_home": 0.15, "p_draw": 0.20, "p_away": 0.65}}
_tha, _taa, _ = kt.compute_tip("A", "B", model_sym, _odds_away)
check(f"Odds (Auswärtsfav) überstimmen sym. Modell ({_tha}:{_taa})", _tha < _taa)
print()

# --- time_weight (Halbwertszeit-Gewichtung, trägt den DC-Fit) ---
print("time_weight:")
from datetime import timedelta
_ref = datetime(2026, 1, 1)
check("t=0 → Gewicht 1.0", abs(kt.time_weight(_ref, _ref) - 1.0) < 1e-9)
check("t=Halbwertszeit → 0.5", abs(kt.time_weight(_ref - timedelta(days=100), _ref, 100) - 0.5) < 1e-6)
check("monoton fallend (älter < neuer)",
      kt.time_weight(_ref - timedelta(days=200), _ref, 100) < kt.time_weight(_ref - timedelta(days=50), _ref, 100))
check("Zukunft geklemmt → 1.0", abs(kt.time_weight(_ref + timedelta(days=30), _ref, 100) - 1.0) < 1e-9)
print()

# --- parse_matches (speist den Split: Endergebnis/Saison/Liga/Datum korrekt, Ungespieltes raus) ---
print("parse_matches:")
_raw = [
    {"group": {"groupOrderID": 7}, "matchDateTimeUTC": "2026-03-01T15:30:00Z",
     "team1": {"teamName": "FC A"}, "team2": {"teamName": "FC B"},
     "matchResults": [{"resultTypeID": 1, "pointsTeam1": 1, "pointsTeam2": 0},
                      {"resultTypeID": 2, "pointsTeam1": 2, "pointsTeam2": 1}]},
    {"group": {"groupOrderID": 8}, "matchDateTimeUTC": "2026-03-08T15:30:00Z",
     "team1": {"teamName": "FC C"}, "team2": {"teamName": "FC D"},
     "matchResults": []},  # noch nicht gespielt → muss rausfallen
]
_pm = kt.parse_matches(_raw, "bl1", 2025)
check("Nur gespielte Matches (1 von 2)", len(_pm) == 1)
check("Endergebnis (resultTypeID==2) genommen",
      _pm[0]["home_goals"] == 2 and _pm[0]["away_goals"] == 1)
check("Saison/Liga/Spieltag korrekt",
      _pm[0]["season"] == 2025 and _pm[0]["league"] == "bl1" and _pm[0]["matchday"] == 7)
check("Datum geparst (Jahr 2026)", _pm[0]["date"].year == 2026)
print()

# --- Ergebnis ---
if errors == 0:
    print("Alle Tests bestanden.")
else:
    print(f"{errors} Test(s) fehlgeschlagen!")
    sys.exit(1)
