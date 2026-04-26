#!/usr/bin/env python3
"""
Kicktipp Bundesliga Predictor
==============================
Dixon-Coles Modell (30%) + Pinnacle-Wettquoten (70%) zur Optimierung
von Tipps gegen das Kicktipp-Punkteschema (1-3 Punkte).

Datenquellen:
  - OpenLigaDB: Ergebnisse 1. + 2. Bundesliga (kostenlos)
  - football-data.co.uk: Historische Pinnacle-Quoten (kostenlos)
  - The Odds API: Live-Quoten vor Spieltag (kostenlos, 500 Credits/Monat)

Verwendung:
  python kicktipp.py predict --season 2025 --matchday 31
  python kicktipp.py backtest --season 2024 --use-odds
  python kicktipp.py backtest --season 2024 --from-matchday 10 --to-matchday 34

Setup:
  pip install -r requirements.txt
  echo "ODDS_API_KEY=dein_key" > .env   # https://the-odds-api.com (kostenlos)
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.optimize import minimize

# .env laden (falls vorhanden)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

CACHE_DIR = Path(__file__).parent / ".cache"

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

OPENLIGADB = "https://api.openligadb.de"
MAX_GOALS = 8          # Ergebnismatrix bis 8:8
MAX_TIP_GOALS = 2      # Tipps maximal bis 2 Tore pro Team
HALF_LIFE_DAYS = 300   # Zeitgewichtung: ~10 Monate Halbwertszeit
MIN_MATCHES = 50       # Mindestanzahl Spiele für Modelltraining
NUM_PREV_SEASONS = 3   # Anzahl Vorsaisons für Training
ODDS_WEIGHT = 0.7      # Mischgewicht Quoten vs. Modell (0=nur Modell, 1=nur Quoten)
LIVE_ODDS_CACHE_HOURS = 6

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281"
ODDS_API_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "soccer_germany_bundesliga"


# ---------------------------------------------------------------------------
# Kicktipp Punkteschema
# ---------------------------------------------------------------------------
# Anpassen an die eigene Kicktipp-Runde:
#   Sieg:           Tendenz richtig → POINTS_TENDENCY
#                   Tordifferenz richtig → POINTS_GOAL_DIFF
#   Unentschieden:  Tendenz richtig → POINTS_DRAW_TENDENCY
#   Exakt:          → POINTS_EXACT
#
# Gängige Kicktipp-Schemata:
#   1-2-3:  Tendenz=1, Diff=2, Remis=2, Exakt=3  (unser Default)
#   2-3-4:  Tendenz=2, Diff=3, Remis=3, Exakt=4  (Kicktipp-Standard)
#   0-2-3:  Tendenz=0, Diff=2, Remis=2, Exakt=3

POINTS_TENDENCY = 1       # Sieg: nur Tendenz richtig
POINTS_GOAL_DIFF = 2      # Sieg: Tordifferenz richtig
POINTS_DRAW_TENDENCY = 2  # Unentschieden: Tendenz richtig (aber nicht exakt)
POINTS_EXACT = 3          # Exaktes Ergebnis


def kicktipp_points(tip_h: int, tip_a: int, real_h: int, real_a: int) -> int:
    """Berechnet Kicktipp-Punkte nach konfiguriertem Schema."""
    if tip_h == real_h and tip_a == real_a:
        return POINTS_EXACT
    tip_diff = tip_h - tip_a
    real_diff = real_h - real_a
    tip_tendency = np.sign(tip_diff)
    real_tendency = np.sign(real_diff)
    if tip_tendency != real_tendency:
        return 0
    # Tendenz stimmt
    if real_h == real_a:
        return POINTS_DRAW_TENDENCY
    if tip_diff == real_diff:
        return POINTS_GOAL_DIFF
    return POINTS_TENDENCY


# ---------------------------------------------------------------------------
# OpenLigaDB Datenabruf
# ---------------------------------------------------------------------------

def fetch_season(season: int, league: str = "bl1") -> list[dict]:
    """Lädt alle Spieltage einer Saison (bl1=1. Liga, bl2=2. Liga), mit Dateicache."""
    label = "1. BL" if league == "bl1" else "2. BL"
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{league}_{season}.json"

    if cache_file.exists():
        print(f"  {label} {season}/{season+1}: Cache", flush=True)
        with open(cache_file) as f:
            return json.load(f)

    url = f"{OPENLIGADB}/getmatchdata/{league}/{season}"
    print(f"  Lade {label} {season}/{season+1}...", end=" ", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    matches = resp.json()
    print(f"{len(matches)} Spiele geladen.")

    with open(cache_file, "w") as f:
        json.dump(matches, f)

    return matches


def parse_matches(raw: list[dict], league: str = "bl1", season: int = 0) -> list[dict]:
    """Parst Rohdaten in einheitliches Format."""
    parsed = []
    for m in raw:
        results = m.get("matchResults", [])
        # Endergebnis bevorzugen (resultTypeID == 2)
        final = next((r for r in results if r["resultTypeID"] == 2), None)
        if final is None:
            continue  # Noch nicht gespielt
        date_str = m.get("matchDateTimeUTC", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            continue
        parsed.append({
            "matchday": m["group"]["groupOrderID"],
            "date": dt,
            "home": m["team1"]["teamName"],
            "away": m["team2"]["teamName"],
            "home_goals": final["pointsTeam1"],
            "away_goals": final["pointsTeam2"],
            "league": league,
            "season": season,
        })
    return parsed


def fetch_matchday_fixtures(season: int, matchday: int) -> list[dict]:
    """Lädt Begegnungen eines einzelnen Spieltags, sortiert nach Anpfiffzeit."""
    url = f"{OPENLIGADB}/getmatchdata/bl1/{season}/{matchday}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    fixtures = []
    for m in raw:
        date_str = m.get("matchDateTimeUTC", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            dt = None
        fixtures.append({
            "matchday": matchday,
            "home": m["team1"]["teamName"],
            "away": m["team2"]["teamName"],
            "kickoff": dt,
        })
    fixtures.sort(key=lambda f: f["kickoff"] or datetime.max.replace(tzinfo=timezone.utc))
    return fixtures


# ---------------------------------------------------------------------------
# football-data.co.uk Quoten (optional)
# ---------------------------------------------------------------------------

def _season_code(season: int) -> str:
    """Konvertiert z.B. 2024 → '2425'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def fetch_odds_csv(season: int) -> list[dict]:
    """Lädt Pinnacle-Quoten + Statistiken von football-data.co.uk, mit Cache."""
    import csv
    import io

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"odds_D1_{season}.csv"

    if cache_file.exists():
        print(f"  Odds {season}/{season+1}: Cache", flush=True)
        content = cache_file.read_text(encoding="utf-8-sig")
    else:
        code = _season_code(season)
        url = f"{FOOTBALL_DATA_URL}/{code}/D1.csv"
        print(f"  Lade Odds {season}/{season+1}...", end=" ", flush=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.text
        cache_file.write_text(content, encoding="utf-8-sig")
        print("OK.")

    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        # Pinnacle-Quoten: PSH (Home), PSD (Draw), PSA (Away)
        try:
            psh = float(row.get("PSH") or row.get("PH", 0))
            psd = float(row.get("PSD") or row.get("PD", 0))
            psa = float(row.get("PSA") or row.get("PA", 0))
        except (ValueError, TypeError):
            psh = psd = psa = 0

        if psh == 0 or psd == 0 or psa == 0:
            continue

        # Quoten → Wahrscheinlichkeiten (Overround entfernen)
        inv_sum = 1/psh + 1/psd + 1/psa
        p_home = (1/psh) / inv_sum
        p_draw = (1/psd) / inv_sum
        p_away = (1/psa) / inv_sum

        entry = {
            "home": row.get("HomeTeam", ""),
            "away": row.get("AwayTeam", ""),
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "home_goals": int(row.get("FTHG", 0)),
            "away_goals": int(row.get("FTAG", 0)),
        }

        # Pinnacle Over/Under 2.5
        try:
            p_ov = float(row.get("P>2.5") or 0)
            p_un = float(row.get("P<2.5") or 0)
            if p_ov > 0 and p_un > 0:
                inv_ou = 1/p_ov + 1/p_un
                entry["p_over"] = (1/p_ov) / inv_ou
                entry["ou_line"] = 2.5
        except (ValueError, TypeError):
            pass

        rows.append(entry)
    return rows


# Mapping: football-data.co.uk Teamnamen → OpenLigaDB Teamnamen
_TEAM_NAME_MAP = {
    "Bayern Munich": "FC Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG Hoffenheim",
    "M'gladbach": "Borussia Mönchengladbach",
    "Monchengladbach": "Borussia Mönchengladbach",
    "Union Berlin": "1. FC Union Berlin",
    "Stuttgart": "VfB Stuttgart",
    "Werder Bremen": "SV Werder Bremen",
    "Mainz": "1. FSV Mainz 05",
    "Augsburg": "FC Augsburg",
    "Bochum": "VfL Bochum",
    "Heidenheim": "1. FC Heidenheim 1846",
    "Darmstadt": "SV Darmstadt 98",
    "Koln": "1. FC Köln",
    "FC Koln": "1. FC Köln",
    "Hertha": "Hertha BSC",
    "Schalke 04": "FC Schalke 04",
    "Greuther Furth": "SpVgg Greuther Fürth",
    "Greuther Fuerth": "SpVgg Greuther Fürth",
    "Arminia": "DSC Arminia Bielefeld",
    "Bielefeld": "DSC Arminia Bielefeld",
    "Holstein Kiel": "Holstein Kiel",
    "St Pauli": "FC St. Pauli",
    "Paderborn": "SC Paderborn 07",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "Nurnberg": "1. FC Nürnberg",
    "Hannover": "Hannover 96",
}


def _normalize_team(name: str) -> str:
    """Normalisiert football-data.co.uk Teamnamen zu OpenLigaDB-Format."""
    return _TEAM_NAME_MAP.get(name, name)


def odds_to_score_matrix(p_home: float, p_draw: float, p_away: float,
                         p_over: float | None = None, ou_line: float = 2.5,
                         max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Konvertiert Quoten-Wahrscheinlichkeiten in eine Score-Matrix.
    Mit Over/Under: 2D-Optimierung (Ratio + Total).
    Ohne: Grid-Search über Gesamttore.
    """
    from scipy.optimize import minimize as scipy_minimize

    def _poisson_matrix(lh, la):
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for gh in range(max_goals + 1):
            for ga in range(max_goals + 1):
                mat[gh, ga] = (math.exp(-lh) * lh**gh / math.factorial(gh) *
                              math.exp(-la) * la**ga / math.factorial(ga))
        return mat

    def _tendency_from_mat(mat):
        ph = np.sum(mat[np.tril_indices(max_goals+1, k=-1)])
        pd = np.trace(mat)
        pa = np.sum(mat[np.triu_indices(max_goals+1, k=1)])
        return ph, pd, pa

    def _over_prob(mat, line):
        """P(home + away > line) aus Score-Matrix."""
        p = 0.0
        for gh in range(max_goals + 1):
            for ga in range(max_goals + 1):
                if gh + ga > line:
                    p += mat[gh, ga]
        return p

    if p_over is not None:
        # 2D-Optimierung: log_ratio und log_total
        def objective(params):
            log_ratio, log_total = params
            total = math.exp(log_total)
            ratio = math.exp(log_ratio)
            lh = total * ratio / (1 + ratio)
            la = total / (1 + ratio)
            mat = _poisson_matrix(lh, la)

            ph, pd, pa = _tendency_from_mat(mat)
            target_hda = np.array([p_home, p_draw, p_away])
            pred_hda = np.clip(np.array([ph, pd, pa]), 1e-8, None)
            kl_hda = np.sum(target_hda * np.log(target_hda / pred_hda))

            # Over/Under Fehler
            pred_over = max(_over_prob(mat, ou_line), 1e-8)
            pred_under = max(1 - pred_over, 1e-8)
            kl_ou = (p_over * math.log(p_over / pred_over) +
                     (1 - p_over) * math.log((1 - p_over) / pred_under))

            return kl_hda + kl_ou

        res = scipy_minimize(objective, [0.0, math.log(2.8)],
                            method="Nelder-Mead", options={"xatol": 1e-6})
        ratio = math.exp(res.x[0])
        total = math.exp(res.x[1])
        lh = total * ratio / (1 + ratio)
        la = total / (1 + ratio)
        best_mat = _poisson_matrix(lh, la)
    else:
        # Fallback: Grid-Search über Gesamttore (wie bisher)
        from scipy.optimize import minimize_scalar
        best_loss = 1e9
        best_mat = None
        for total in [2.2, 2.4, 2.6, 2.8, 3.0, 3.2]:
            def obj(log_ratio, tot=total):
                ratio = math.exp(log_ratio)
                lh = tot * ratio / (1 + ratio)
                la = tot / (1 + ratio)
                mat = _poisson_matrix(lh, la)
                ph, pd, pa = _tendency_from_mat(mat)
                target = np.array([p_home, p_draw, p_away])
                pred = np.clip(np.array([ph, pd, pa]), 1e-8, None)
                return np.sum(target * np.log(target / pred))

            res = minimize_scalar(obj, bounds=(-2, 2), method="bounded")
            if res.fun < best_loss:
                best_loss = res.fun
                ratio = math.exp(res.x)
                lh = total * ratio / (1 + ratio)
                la = total / (1 + ratio)
                best_mat = _poisson_matrix(lh, la)

    best_mat /= best_mat.sum()
    return best_mat


def fetch_live_odds() -> dict:
    """
    Holt aktuelle Quoten von The Odds API (kostenlos, 500 Req/Monat).
    Braucht ODDS_API_KEY Umgebungsvariable.
    Gibt dict {(home, away): {"p_home", "p_draw", "p_away"}} zurück.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {}

    # Cache: max 6 Stunden alt, dann neu laden
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "live_odds.json"
    max_age = LIVE_ODDS_CACHE_HOURS * 3600

    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < max_age:
            print(f"  Live-Quoten: Cache ({age/3600:.1f}h alt)", flush=True)
            with open(cache_file) as f:
                data = json.load(f)
        else:
            data = None
    else:
        data = None

    if data is None:
        url = f"{ODDS_API_URL}/sports/{ODDS_API_SPORT}/odds/"
        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h,totals",
        }
        print("  Lade Live-Quoten (The Odds API)...", end=" ", flush=True)
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"Fehler: {e}")
            return {}

        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"{len(data)} Spiele. (Credits übrig: {remaining})")

        with open(cache_file, "w") as f:
            json.dump(data, f)

    odds_dict = {}
    for event in data:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # Suche Pinnacle-Quoten (Fallback: erster Bookmaker)
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        # H2H-Quoten
        h2h = None
        for bm in [next((b for b in bookmakers if b["key"] == "pinnacle"), None)] + bookmakers:
            if bm is None:
                continue
            h2h = next((m for m in bm.get("markets", []) if m["key"] == "h2h"), None)
            if h2h:
                break
        if not h2h:
            continue

        outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
        price_h = outcomes.get(home_team, 0)
        price_d = outcomes.get("Draw", 0)
        price_a = outcomes.get(away_team, 0)

        if price_h == 0 or price_d == 0 or price_a == 0:
            continue

        # Overround entfernen
        inv_sum = 1/price_h + 1/price_d + 1/price_a
        p_home = (1/price_h) / inv_sum
        p_draw = (1/price_d) / inv_sum
        p_away = (1/price_a) / inv_sum

        # Over/Under-Quoten suchen (aus allen Bookmakers)
        p_over = None
        ou_line = None
        for bm in bookmakers:
            totals = next((m for m in bm.get("markets", []) if m["key"] == "totals"), None)
            if totals:
                for o in totals["outcomes"]:
                    if o["name"] == "Over":
                        over_price = o["price"]
                        point = o.get("point", 2.5)
                        under_price = next(
                            (x["price"] for x in totals["outcomes"] if x["name"] == "Under"), 0)
                        if over_price > 0 and under_price > 0:
                            inv = 1/over_price + 1/under_price
                            p_over = (1/over_price) / inv
                            ou_line = point
                            break
            if p_over is not None:
                break

        # Team-Name normalisieren (Odds API verwendet englische Namen)
        home_norm = _normalize_team(home_team)
        away_norm = _normalize_team(away_team)
        # O/U wird abgerufen aber nicht genutzt (CV zeigt keine Verbesserung)
        odds_dict[(home_norm, away_norm)] = {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        }

    return odds_dict


# Odds API Teamnamen-Mapping (zusätzlich zu football-data.co.uk)
_TEAM_NAME_MAP.update({
    "Borussia Dortmund": "Borussia Dortmund",
    "Bayern Munich": "FC Bayern München",
    "Bayer 04 Leverkusen": "Bayer 04 Leverkusen",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "VfB Stuttgart": "VfB Stuttgart",
    "SC Freiburg": "SC Freiburg",
    "VfL Wolfsburg": "VfL Wolfsburg",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Borussia Mönchengladbach": "Borussia Mönchengladbach",
    "TSG 1899 Hoffenheim": "TSG Hoffenheim",
    "TSG Hoffenheim": "TSG Hoffenheim",
    "1. FC Union Berlin": "1. FC Union Berlin",
    "1. FSV Mainz 05": "1. FSV Mainz 05",
    "FC Augsburg": "FC Augsburg",
    "SV Werder Bremen": "SV Werder Bremen",
    "VfL Bochum": "VfL Bochum",
    "1. FC Heidenheim 1846": "1. FC Heidenheim 1846",
    "FC St. Pauli": "FC St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "RB Leipzig": "RB Leipzig",
    "FC Bayern Munich": "FC Bayern München",
    "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "FSV Mainz 05": "1. FSV Mainz 05",
    "1. FC Heidenheim": "1. FC Heidenheim 1846",
    "Hamburger SV": "Hamburger SV",
})


# ---------------------------------------------------------------------------
# Dixon-Coles Modell
# ---------------------------------------------------------------------------

def time_weight(match_date: datetime, ref_date: datetime, half_life_days: float | None = None) -> float:
    """Exponentielles Zeitgewicht – neuere Spiele zählen mehr."""
    if half_life_days is None:
        half_life_days = HALF_LIFE_DAYS
    delta = (ref_date - match_date).total_seconds() / 86400
    delta = max(delta, 0)
    return math.exp(-math.log(2) * delta / half_life_days)


def dc_rho_correction(goals_h: int, goals_a: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles Korrekturfaktor für niedrige Scores."""
    if goals_h == 0 and goals_a == 0:
        return 1 - lambda_h * lambda_a * rho
    elif goals_h == 1 and goals_a == 0:
        return 1 + lambda_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1 + lambda_h * rho
    elif goals_h == 1 and goals_a == 1:
        return 1 - rho
    return 1.0


def build_team_index(matches: list[dict]) -> dict[str, int]:
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    return {t: i for i, t in enumerate(teams)}


def _precompute_match_arrays(matches: list[dict], team_idx: dict):
    """Vorberechnung der Match-Arrays für vektorisierte Likelihood."""
    hi = np.array([team_idx[m["home"]] for m in matches])
    ai = np.array([team_idx[m["away"]] for m in matches])
    gh = np.array([m["home_goals"] for m in matches], dtype=np.float64)
    ga = np.array([m["away_goals"] for m in matches], dtype=np.float64)
    lgamma_gh = np.array([math.lgamma(g + 1) for g in gh])
    lgamma_ga = np.array([math.lgamma(g + 1) for g in ga])
    return hi, ai, gh, ga, lgamma_gh, lgamma_ga


def neg_log_likelihood(params: np.ndarray, match_arrays: tuple, n: int,
                       weights: np.ndarray) -> float:
    attack = params[:n]
    defense = params[n:2*n]
    home_adv = params[2*n]
    rho = params[2*n + 1]

    hi, ai, gh, ga, lgamma_gh, lgamma_ga = match_arrays

    lh = np.exp(attack[hi] - defense[ai] + home_adv)
    la = np.exp(attack[ai] - defense[hi])

    # Poisson log-likelihood (vektorisiert)
    log_lh = np.log(lh)
    log_la = np.log(la)
    ll = (gh * log_lh - lh - lgamma_gh +
          ga * log_la - la - lgamma_ga)

    # Dixon-Coles Korrektur (vektorisiert)
    tau = np.ones(len(gh))
    m00 = (gh == 0) & (ga == 0)
    m10 = (gh == 1) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m11 = (gh == 1) & (ga == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m11] = 1 - rho

    if np.any(tau <= 0):
        return 1e9

    ll += np.log(tau)

    return -np.dot(weights, ll)


def fit_dixon_coles(matches: list[dict], ref_date: datetime) -> dict:
    """Schätzt Dixon-Coles Parameter via MLE."""
    team_idx = build_team_index(matches)
    n = len(team_idx)
    weights = np.array([time_weight(m["date"], ref_date) for m in matches])
    match_arrays = _precompute_match_arrays(matches, team_idx)

    # Startwerte
    x0 = np.zeros(2 * n + 2)
    x0[2*n] = 0.3   # Heimvorteil
    x0[2*n+1] = -0.1  # rho

    # Identifikation: attack[0] = 0 fixieren
    def objective(p):
        p_full = np.concatenate([[0.0], p[:(n-1)], p[(n-1):]])
        return neg_log_likelihood(p_full, match_arrays, n, weights)

    x0_free = np.concatenate([x0[1:n], x0[n:]])
    result = minimize(objective, x0_free, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-9})

    p_full = np.concatenate([[0.0], result.x[:(n-1)], result.x[(n-1):]])
    attack = {t: p_full[i] for t, i in team_idx.items()}
    defense = {t: p_full[n + i] for t, i in team_idx.items()}

    return {
        "attack": attack,
        "defense": defense,
        "home_adv": p_full[2*n],
        "rho": p_full[2*n+1],
        "teams": list(team_idx.keys()),
    }


# ---------------------------------------------------------------------------
# Ergebnismatrix & Tipp-Optimierung
# ---------------------------------------------------------------------------

def score_matrix(home: str, away: str, model: dict, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Berechnet P(home_goals=i, away_goals=j) für alle i,j."""
    lh = math.exp(model["attack"][home] - model["defense"][away] + model["home_adv"])
    la = math.exp(model["attack"][away] - model["defense"][home])
    rho = model["rho"]

    mat = np.zeros((max_goals + 1, max_goals + 1))
    for gh in range(max_goals + 1):
        for ga in range(max_goals + 1):
            p = (math.exp(-lh) * lh**gh / math.factorial(gh) *
                 math.exp(-la) * la**ga / math.factorial(ga))
            p *= dc_rho_correction(gh, ga, lh, la, rho)
            mat[gh, ga] = max(p, 0)

    mat /= mat.sum()
    return mat


def _build_points_table(max_goals: int = MAX_GOALS) -> np.ndarray:
    """Vorberechnung: points[th, ta, rh, ra] = Kicktipp-Punkte."""
    n = max_goals + 1
    table = np.zeros((n, n, n, n), dtype=np.float64)
    for th in range(n):
        for ta in range(n):
            for rh in range(n):
                for ra in range(n):
                    table[th, ta, rh, ra] = kicktipp_points(th, ta, rh, ra)
    return table

_POINTS_TABLE = _build_points_table()


def best_tip(home: str, away: str, model: dict) -> tuple[int, int, float]:
    """
    Wählt den Tipp mit dem höchsten erwarteten Kicktipp-Punktewert.
    Gibt (tip_h, tip_a, expected_points) zurück.
    """
    mat = score_matrix(home, away, model)
    # ev[th, ta] = sum over rh,ra of mat[rh,ra] * points[th,ta,rh,ra]
    ev = np.einsum("ra,tpra->tp", mat, _POINTS_TABLE)
    # Nur Tipps bis MAX_TIP_GOALS berücksichtigen
    ev_clipped = ev[:MAX_TIP_GOALS + 1, :MAX_TIP_GOALS + 1]
    idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
    return idx[0], idx[1], ev_clipped[idx]


def best_tip_combined(home: str, away: str, model: dict,
                      odds_mat: np.ndarray, weight_odds: float = ODDS_WEIGHT) -> tuple[int, int, float]:
    """
    Kombiniert Dixon-Coles und Quoten-basierte Score-Matrix.
    weight_odds=0: nur Modell, weight_odds=1: nur Quoten.
    """
    dc_mat = score_matrix(home, away, model)
    # Gewichtete Mischung der beiden Matrizen
    combined = (1 - weight_odds) * dc_mat + weight_odds * odds_mat
    combined /= combined.sum()
    ev = np.einsum("ra,tpra->tp", combined, _POINTS_TABLE)
    ev_clipped = ev[:MAX_TIP_GOALS + 1, :MAX_TIP_GOALS + 1]
    idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
    return idx[0], idx[1], ev_clipped[idx]


def _find_odds(odds_dict: dict, home: str, away: str) -> dict | None:
    """Sucht Quoten für ein Fixture — auch bei vertauschten Teams."""
    if not odds_dict:
        return None
    # Exakter Match
    if (home, away) in odds_dict:
        return odds_dict[(home, away)]
    # Umgekehrte Paarung (Hin-/Rückspiel) → Wahrscheinlichkeiten tauschen
    if (away, home) in odds_dict:
        od = odds_dict[(away, home)]
        return {"p_home": od["p_away"], "p_draw": od["p_draw"], "p_away": od["p_home"]}
    return None


# HINWEIS: Die Recalibration-Korrektur wurde getestet und führt zu
# -27 bis -35 Punkten gegenüber dem Baseline-Modell. Bias-Muster sind
# zwischen Saisons nicht stabil genug, um sie post-hoc auszunutzen.
# Funktion bleibt für zukünftige Experimente mit größeren Datensätzen.
# Siehe README → "Was wir getestet haben (und was nicht hilft)".

def recalibrate_score_matrix(mat: np.ndarray, correction_table: np.ndarray) -> np.ndarray:
    """Korrigiert eine Score-Matrix elementweise und renormiert."""
    n = min(mat.shape[0], correction_table.shape[0])
    corrected = mat.copy()
    corrected[:n, :n] *= correction_table[:n, :n]
    corrected = np.clip(corrected, 0, None)
    corrected /= corrected.sum()
    return corrected


def fit_recalibration(train_seasons: list[int], min_obs: int = 20) -> np.ndarray:
    """Lernt eine Korrekturtabelle aus vergangenen Saisons (kein Data Leakage).

    Für jedes Score (h,a): correction = actual_freq / predicted_freq.
    Scores mit < min_obs Beobachtungen werden auf 1.0 (keine Korrektur) gesetzt.
    """
    n = MAX_GOALS + 1
    predicted_sum = np.zeros((n, n))
    actual_count = np.zeros((n, n))
    total_matches = 0

    for season in train_seasons:
        all_matches = load_all_matches(season)
        season_matches = [m for m in all_matches if
                          m["league"] == "bl1" and m["season"] == season]

        odds_data = {}
        try:
            odds_rows = fetch_odds_csv(season)
            for row in odds_rows:
                key = (_normalize_team(row["home"]), _normalize_team(row["away"]))
                odds_data[key] = {k: v for k, v in row.items()}
        except Exception:
            pass

        for md in range(1, 35):
            cutoff = [m for m in season_matches if m["matchday"] < md]
            prev = [m for m in all_matches if not (m["league"] == "bl1" and m["season"] == season)]
            training = prev + cutoff
            if len(training) < MIN_MATCHES:
                continue

            md_matches = [m for m in season_matches if m["matchday"] == md]
            if not md_matches:
                continue
            ref_date = min(m["date"] for m in md_matches)
            model = fit_dixon_coles(training, ref_date)

            for m in md_matches:
                home, away = m["home"], m["away"]
                if home not in model["attack"] or away not in model["attack"]:
                    continue

                # Score-Matrix berechnen (gleiche Logik wie compute_tip)
                od = _find_odds(odds_data, _normalize_team(home),
                                _normalize_team(away)) if odds_data else None
                if od:
                    dc_mat = score_matrix(home, away, model)
                    o_mat = odds_to_score_matrix(
                        od["p_home"], od["p_draw"], od["p_away"],
                        od.get("p_over"), od.get("ou_line", 2.5))
                    mat = (1 - ODDS_WEIGHT) * dc_mat + ODDS_WEIGHT * o_mat
                    mat /= mat.sum()
                else:
                    mat = score_matrix(home, away, model)

                predicted_sum += mat
                rh = min(m["home_goals"], n - 1)
                ra = min(m["away_goals"], n - 1)
                actual_count[rh, ra] += 1
                total_matches += 1

    if total_matches == 0:
        return np.ones((n, n))

    predicted_freq = predicted_sum / total_matches
    actual_freq = actual_count / total_matches

    # Korrekturfaktor mit Smoothing
    correction = np.ones((n, n))
    for h in range(n):
        for a in range(n):
            if actual_count[h, a] >= min_obs and predicted_freq[h, a] > 1e-6:
                correction[h, a] = actual_freq[h, a] / predicted_freq[h, a]

    return correction


def compute_tip(home: str, away: str, model: dict, odds_dict: dict = None,
                correction_table: np.ndarray = None):
    """Berechnet den optimalen Tipp — mit Quoten und optionaler Recalibration."""
    od = _find_odds(odds_dict, home, away) if odds_dict else None
    if od:
        odds_mat = odds_to_score_matrix(od["p_home"], od["p_draw"], od["p_away"],
                                        od.get("p_over"), od.get("ou_line", 2.5))
        dc_mat = score_matrix(home, away, model)
        combined = (1 - ODDS_WEIGHT) * dc_mat + ODDS_WEIGHT * odds_mat
        combined /= combined.sum()
    else:
        combined = score_matrix(home, away, model)

    if correction_table is not None:
        combined = recalibrate_score_matrix(combined, correction_table)

    ev = np.einsum("ra,tpra->tp", combined, _POINTS_TABLE)
    ev_clipped = ev[:MAX_TIP_GOALS + 1, :MAX_TIP_GOALS + 1]
    idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
    return idx[0], idx[1], ev_clipped[idx]


def load_all_matches(season: int) -> list[dict]:
    """Lädt Trainings-Matches: aktuelle + Vorsaisons, jeweils 1. und 2. Liga."""
    all_matches = []
    for s in range(season - NUM_PREV_SEASONS, season + 1):
        for league in ["bl1", "bl2"]:
            try:
                raw = fetch_season(s, league)
                all_matches += parse_matches(raw, league, s)
            except Exception as e:
                print(f"  Warnung: {league} Saison {s} nicht geladen ({e})")
    return all_matches


def tendency_str(h: int, a: int) -> str:
    if h > a:
        return "Heimsieg"
    elif h < a:
        return "Auswärtssieg"
    return "Unentschieden"


# ---------------------------------------------------------------------------
# CLI: predict
# ---------------------------------------------------------------------------

def cmd_predict(args):
    use_odds = getattr(args, "use_odds", False) or bool(os.environ.get("ODDS_API_KEY"))
    mode_str = " [+Odds]" if use_odds else ""
    print(f"\n=== PROGNOSE Spieltag {args.matchday}, Saison {args.season}/{args.season+1}{mode_str} ===\n")

    all_matches = load_all_matches(args.season)

    # Nur Spiele vor aktuellem Spieltag dieser Saison als Training
    training = [m for m in all_matches
                if not (m["matchday"] >= args.matchday and
                        m["date"].year >= args.season)]

    if len(training) < MIN_MATCHES:
        print(f"Fehler: Nur {len(training)} Trainingsmatches – zu wenig.")
        sys.exit(1)

    print(f"\nTrainiere Dixon-Coles Modell auf {len(training)} Spielen...")
    model = fit_dixon_coles(training, datetime.now(tz=timezone.utc))

    # Live-Quoten holen (optional)
    live_odds = {}
    if use_odds:
        live_odds = fetch_live_odds()
        if not live_odds:
            print("  Hinweis: Keine Live-Quoten verfügbar, nutze nur Modell.")

    # Fixtures laden
    try:
        fixtures = fetch_matchday_fixtures(args.season, args.matchday)
    except Exception as e:
        print(f"Fehler beim Laden der Begegnungen: {e}")
        sys.exit(1)

    # Warnung bei ungematchten Quoten
    if live_odds:
        matched = sum(1 for f in fixtures
                      if _find_odds(live_odds, f["home"], f["away"]) is not None)
        if matched == 0 and fixtures:
            print("  ⚠ Keine Quoten konnten zugeordnet werden!")
            print("    Prüfe _TEAM_NAME_MAP in kicktipp.py (Aufsteiger?)")
            print(f"    Fixtures: {[f['home'] for f in fixtures[:3]]}...")
            odds_teams = list(live_odds.keys())[:3]
            print(f"    Odds-API: {[h for h, a in odds_teams]}...")
            print()

    print(f"\n{'Begegnung':<52} {'Tipp':>8}  {'E[Pkt]':>7}  Tendenz")
    print("-" * 85)
    for f in fixtures:
        home, away = f["home"], f["away"]
        if home not in model["attack"] or away not in model["attack"]:
            print(f"  {home} vs {away}: Team unbekannt")
            continue

        th, ta, ev = compute_tip(home, away, model, live_odds or None)
        has_odds = _find_odds(live_odds, home, away) is not None
        marker = "⚡" if has_odds else " "
        label = f"{home} – {away}"
        print(f"{marker} {label:<50} {th}:{ta}  {ev:>6.3f}  {tendency_str(th, ta)}")

    print("  ⚡ = mit Live-Quoten" if live_odds else "")
    print()


# ---------------------------------------------------------------------------
# Kalibrierungs-Analyse
# ---------------------------------------------------------------------------

_CALIBRATION_SCORES = [
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),
]

_CALIBRATION_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 100)]


def _tendency_probs(mat: np.ndarray) -> tuple[float, float, float]:
    """Marginalisiert die Score-Matrix zu (P_home, P_draw, P_away)."""
    n = mat.shape[0]
    p_home = sum(mat[h, a] for h in range(n) for a in range(n) if h > a)
    p_draw = sum(mat[h, h] for h in range(n))
    p_away = sum(mat[h, a] for h in range(n) for a in range(n) if h < a)
    return p_home, p_draw, p_away


def cmd_calibration(args):
    season = args.season
    from_md = args.from_matchday
    to_md = args.to_matchday
    use_odds = getattr(args, "use_odds", False)

    print(f"\n=== KALIBRIERUNGS-ANALYSE Saison {season}/{season+1}, "
          f"Spieltage {from_md}–{to_md}"
          f"{' [+Odds]' if use_odds else ''} ===\n")

    all_matches = load_all_matches(season)
    season_matches = [m for m in all_matches if
                      m["league"] == "bl1" and m["season"] == season]

    odds_data = {}
    if use_odds:
        try:
            odds_rows = fetch_odds_csv(season)
            for row in odds_rows:
                key = (_normalize_team(row["home"]), _normalize_team(row["away"]))
                odds_data[key] = {k: v for k, v in row.items()}
        except Exception as e:
            print(f"  Warnung: Quoten nicht geladen ({e})")

    # Pro Spiel: Modell-Score-Matrix berechnen
    predictions = []
    skipped = 0

    for md in range(from_md, to_md + 1):
        cutoff = [m for m in season_matches if m["matchday"] < md]
        prev = [m for m in all_matches if m not in season_matches]
        training = prev + cutoff
        if len(training) < MIN_MATCHES:
            continue

        md_matches = [m for m in season_matches if m["matchday"] == md]
        if not md_matches:
            continue

        ref_date = min(m["date"] for m in md_matches)
        model = fit_dixon_coles(training, ref_date)

        for m in md_matches:
            home, away = m["home"], m["away"]
            if home not in model["attack"] or away not in model["attack"]:
                skipped += 1
                continue

            od = _find_odds(odds_data, _normalize_team(home),
                            _normalize_team(away)) if odds_data else None
            if od:
                dc_mat = score_matrix(home, away, model)
                odds_mat = odds_to_score_matrix(
                    od["p_home"], od["p_draw"], od["p_away"],
                    od.get("p_over"), od.get("ou_line", 2.5)
                )
                mat = (1 - ODDS_WEIGHT) * dc_mat + ODDS_WEIGHT * odds_mat
                mat /= mat.sum()
            else:
                mat = score_matrix(home, away, model)

            predictions.append((mat, m["home_goals"], m["away_goals"]))

    n = len(predictions)
    if n == 0:
        print("  Keine auswertbaren Spiele gefunden.")
        return
    print(f"  Spiele: {n} (übersprungen: {skipped})\n")

    # ---- 1X2-Kalibrierung ----
    print(f"  [1X2-Tendenz] Reliability:\n")
    print(f"  {'Bin':>20s}   {'n':>5s}   {'vorhergesagt':>12s}   {'tatsächlich':>11s}   {'Diff':>7s}")
    print(f"  {'-'*65}")

    for outcome_idx, outcome_name in [(0, "Heimsieg"), (1, "Remis"), (2, "Auswärtssieg")]:
        print(f"\n  {outcome_name}:")
        for low, high in _CALIBRATION_BINS:
            in_bin = []
            for mat, rh, ra in predictions:
                p_h, p_d, p_a = _tendency_probs(mat)
                p_pred = [p_h, p_d, p_a][outcome_idx]
                if low / 100 <= p_pred < high / 100:
                    actual = 0
                    if outcome_idx == 0 and rh > ra:
                        actual = 1
                    elif outcome_idx == 1 and rh == ra:
                        actual = 1
                    elif outcome_idx == 2 and rh < ra:
                        actual = 1
                    in_bin.append((p_pred, actual))

            if not in_bin:
                continue
            avg_pred = np.mean([x[0] for x in in_bin])
            avg_actual = np.mean([x[1] for x in in_bin])
            diff = avg_actual - avg_pred
            sign = "+" if diff > 0 else ""
            marker = "▲" if diff > 0.02 else ("▼" if diff < -0.02 else "·")
            print(f"  {f'{low:2d}-{high:3d}%':>20s}   {len(in_bin):>5d}   "
                  f"{avg_pred*100:>10.1f}%   {avg_actual*100:>9.1f}%   "
                  f"{sign}{diff*100:>5.1f}pp {marker}")

    # ---- Score-Kalibrierung ----
    print(f"\n\n  [Einzelergebnisse] Reliability:\n")
    print(f"  {'Score':>6s}   {'Modell-Ø':>10s}   {'Tatsächlich':>12s}   "
          f"{'Diff':>7s}   {'Bewertung':>15s}")
    print(f"  {'-'*60}")

    for (h, a) in _CALIBRATION_SCORES:
        avg_pred = np.mean([mat[h, a] for mat, _, _ in predictions])
        actual = np.mean([1 if (rh == h and ra == a) else 0
                          for _, rh, ra in predictions])
        diff = actual - avg_pred
        sign = "+" if diff > 0 else ""
        if abs(diff) < 0.005:
            verdict = "gut"
        elif abs(diff) < 0.015:
            verdict = "okay"
        else:
            verdict = "Abweichung"
        print(f"  {h}:{a:<4d}   {avg_pred*100:>8.2f}%   {actual*100:>10.2f}%   "
              f"{sign}{diff*100:>5.2f}pp   {verdict:>15s}")

    # ---- Globale Metriken ----
    print(f"\n\n  [Globale Metriken]\n")

    brier_sum = 0.0
    log_loss_sum = 0.0
    for mat, rh, ra in predictions:
        p_h, p_d, p_a = _tendency_probs(mat)
        if rh > ra:
            actual = (1, 0, 0)
        elif rh == ra:
            actual = (0, 1, 0)
        else:
            actual = (0, 0, 1)
        brier_sum += sum((p - a) ** 2 for p, a in zip([p_h, p_d, p_a], actual))
        actual_p = [p_h, p_d, p_a][actual.index(1)]
        log_loss_sum += -math.log(max(actual_p, 1e-12))

    brier = brier_sum / n
    log_loss = log_loss_sum / n
    brier_uninformed = 2 / 3
    log_loss_uninformed = math.log(3)

    print(f"    Brier-Score:        {brier:.4f}    "
          f"(uninformiert: {brier_uninformed:.4f}, niedriger = besser)")
    print(f"    Log-Loss:           {log_loss:.4f}    "
          f"(uninformiert: {log_loss_uninformed:.4f})")

    skill_brier = 1 - brier / brier_uninformed
    skill_ll = 1 - log_loss / log_loss_uninformed
    print(f"    Brier Skill Score:  {skill_brier:.3f}    "
          f"(0 = uninformiert, 1 = perfekt)")
    print(f"    Log-Loss Skill:     {skill_ll:.3f}\n")


# ---------------------------------------------------------------------------
# CLI: backtest
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ceiling-Analyse
# ---------------------------------------------------------------------------
#
# Drei verschiedene Definitionen des Ceilings:
#
#   market    Ceiling 2: Optimaler EV-Tipp gegen die aus Pinnacle-Closing-Odds
#             geschätzte Wahrscheinlichkeitsverteilung P*[h, a]. Das ist das
#             realistische theoretische Maximum für ein Modell, das die wahre
#             Verteilung kennt.
#
#   hindsight Ceiling 3: Optimaler Tipp pro Spiel mit perfekter Hindsight,
#             aber begrenzt auf MAX_TIP_GOALS. Überschätzt das echte Ceiling
#             stark — zeigt aber den irreduziblen Stochastik-Anteil
#             (Differenz zu market).
#
#   bins      Constrained Hindsight: ein einziger Tipp pro Quoten-Bin
#             (z.B. "klarer Heimsieg") über alle Spiele in dem Bin.
#             Realistischer als hindsight, weil ein Modell Spiele ohnehin
#             nur grob klassifizieren kann.

def _bin_for_match(p_home: float, p_draw: float, p_away: float) -> str:
    """Klassifiziert ein Spiel anhand der Quoten-Wahrscheinlichkeiten."""
    if p_home > 0.6:
        return "klarer_heimsieg"
    if p_away > 0.5:
        return "klarer_auswaertssieg"
    if p_home > 0.45:
        return "heimfavorit"
    if p_away > 0.35:
        return "auswaertsfavorit"
    return "ausgeglichen"


def _ev_market(p_star: np.ndarray) -> tuple[int, int, float]:
    """Maximalen EV-Tipp gegen Verteilung P* finden (begrenzt auf MAX_TIP_GOALS)."""
    ev = np.einsum("ra,tpra->tp", p_star, _POINTS_TABLE)
    ev_clipped = ev[:MAX_TIP_GOALS + 1, :MAX_TIP_GOALS + 1]
    idx = np.unravel_index(ev_clipped.argmax(), ev_clipped.shape)
    return idx[0], idx[1], ev_clipped[idx]


def _hindsight_best_tip(real_h: int, real_a: int) -> tuple[int, int, int]:
    """Bester Tipp bei perfekter Hindsight, begrenzt auf MAX_TIP_GOALS."""
    best = (0, 0, -1)
    for th in range(MAX_TIP_GOALS + 1):
        for ta in range(MAX_TIP_GOALS + 1):
            pts = kicktipp_points(th, ta, real_h, real_a)
            if pts > best[2]:
                best = (th, ta, pts)
    return best


def cmd_ceiling(args):
    season = args.season
    from_md = args.from_matchday
    to_md = args.to_matchday

    print(f"\n=== CEILING-ANALYSE Saison {season}/{season+1}, "
          f"Spieltage {from_md}–{to_md} ===\n")

    # Daten laden
    all_matches = load_all_matches(season)
    season_matches = [m for m in all_matches if
                      m["league"] == "bl1" and m["season"] == season]
    season_matches = [m for m in season_matches
                      if from_md <= m["matchday"] <= to_md]

    # Quoten laden (für market-Ceiling Pflicht, sonst optional)
    odds_data = {}
    try:
        odds_rows = fetch_odds_csv(season)
        for row in odds_rows:
            key = (_normalize_team(row["home"]), _normalize_team(row["away"]))
            odds_data[key] = {k: v for k, v in row.items()}
    except Exception as e:
        print(f"  Warnung: Quoten nicht geladen ({e})")
        if "market" in args.modes or "bins" in args.modes:
            print("  → market/bins-Ceiling brauchen Quoten, breche ab.")
            return

    # Pro Spiel: P* schätzen (aus Quoten) und Ergebnis merken
    matches_with_data = []
    skipped_no_odds = 0
    for m in season_matches:
        od = _find_odds(odds_data,
                        _normalize_team(m["home"]),
                        _normalize_team(m["away"]))
        if not od:
            skipped_no_odds += 1
            continue
        p_star = odds_to_score_matrix(
            od["p_home"], od["p_draw"], od["p_away"],
            od.get("p_over"), od.get("ou_line", 2.5)
        )
        matches_with_data.append({
            "match": m,
            "odds": od,
            "p_star": p_star,
        })

    n_total = len(matches_with_data)
    if n_total == 0:
        print("  Keine Spiele mit Quoten gefunden.")
        return
    print(f"  Spiele insgesamt: {n_total}"
          f" (übersprungen wegen fehlender Quoten: {skipped_no_odds})\n")

    # ---- Ceiling 2: Market ----
    if "market" in args.modes:
        total_ev = 0.0
        for entry in matches_with_data:
            _, _, ev = _ev_market(entry["p_star"])
            total_ev += ev
        ev_per_match = total_ev / n_total
        print(f"  [market]    Ceiling 2 (EV gegen P* aus Pinnacle-Odds):")
        print(f"              {total_ev:7.2f} Pkt insgesamt, "
              f"Ø {ev_per_match:.3f} / Spiel")

    # ---- Ceiling 3: Hindsight ----
    if "hindsight" in args.modes:
        total_pts = 0
        for entry in matches_with_data:
            m = entry["match"]
            _, _, pts = _hindsight_best_tip(m["home_goals"], m["away_goals"])
            total_pts += pts
        avg = total_pts / n_total
        print(f"  [hindsight] Ceiling 3 (perfekte Hindsight, MAX_TIP_GOALS={MAX_TIP_GOALS}):")
        print(f"              {total_pts:7d} Pkt insgesamt, "
              f"Ø {avg:.3f} / Spiel")

    # ---- Ceiling 3b: Constrained Hindsight (Bins) ----
    if "bins" in args.modes:
        # Spiele in Bins gruppieren
        bins: dict[str, list] = {}
        for entry in matches_with_data:
            od = entry["odds"]
            label = _bin_for_match(od["p_home"], od["p_draw"], od["p_away"])
            bins.setdefault(label, []).append(entry)

        print(f"  [bins]      Constrained Hindsight (ein Tipp pro Quoten-Bin):")
        total_pts = 0
        for label, entries in sorted(bins.items()):
            # Für jeden Kandidaten-Tipp die Punkte über alle Spiele im Bin summieren
            best_total = -1
            best_tip = (0, 0)
            for th in range(MAX_TIP_GOALS + 1):
                for ta in range(MAX_TIP_GOALS + 1):
                    s = sum(kicktipp_points(th, ta,
                                            e["match"]["home_goals"],
                                            e["match"]["away_goals"])
                            for e in entries)
                    if s > best_total:
                        best_total = s
                        best_tip = (th, ta)
            total_pts += best_total
            avg = best_total / len(entries)
            print(f"                {label:22s} n={len(entries):3d}  "
                  f"bester Tipp {best_tip[0]}:{best_tip[1]}  "
                  f"→ {best_total:4d} Pkt (Ø {avg:.3f})")
        avg = total_pts / n_total
        print(f"              Summe: {total_pts:7d} Pkt, Ø {avg:.3f} / Spiel")

    # ---- Vergleich: tatsächliche Modell-Performance ----
    if args.compare_model:
        print(f"\n  Vergleich: Modell-Performance (mit Quoten) auf gleichem Sample:")
        # Vollständiger Backtest-Loop, aber nur für die ausgewählten Spiele
        total_pts = 0
        for entry in matches_with_data:
            m = entry["match"]
            md = m["matchday"]
            cutoff = [x for x in season_matches if x["matchday"] < md]
            prev = [x for x in all_matches if x not in season_matches]
            training = prev + cutoff
            if len(training) < MIN_MATCHES:
                continue
            ref_date = m["date"]
            model = fit_dixon_coles(training, ref_date)
            if m["home"] not in model["attack"] or m["away"] not in model["attack"]:
                continue
            th, ta, _ = compute_tip(m["home"], m["away"], model, odds_data)
            total_pts += kicktipp_points(th, ta, m["home_goals"], m["away_goals"])
        avg = total_pts / n_total
        print(f"              {total_pts:7d} Pkt insgesamt, "
              f"Ø {avg:.3f} / Spiel")

    print()


def cmd_backtest(args):
    season = args.season
    from_md = args.from_matchday
    to_md = args.to_matchday

    use_odds = getattr(args, "use_odds", False)
    mode_str = " [+Odds]" if use_odds else ""
    print(f"\n=== BACKTESTING Saison {season}/{season+1}, Spieltage {from_md}–{to_md}{mode_str} ===\n")

    all_matches = load_all_matches(season)

    # Quoten laden (optional)
    odds_data = {}
    if use_odds:
        try:
            odds_rows = fetch_odds_csv(season)
            for row in odds_rows:
                key = (_normalize_team(row["home"]), _normalize_team(row["away"]))
                odds_data[key] = {k: v for k, v in row.items()
                                  if k not in ("p_over", "ou_line")}
        except Exception as e:
            print(f"  Warnung: Quoten nicht geladen ({e})")

    # Nur 1. Liga der aktuellen Saison für Auswertung
    season_matches = [m for m in all_matches if
                      m["league"] == "bl1" and m["season"] == season]

    total_pts = 0
    total_games = 0
    results_by_md = []
    pts_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

    for md in range(from_md, to_md + 1):
        # Training: alles VOR diesem Spieltag
        cutoff_matches = [m for m in season_matches if m["matchday"] < md]
        prev_season = [m for m in all_matches if m not in season_matches]
        training = prev_season + cutoff_matches

        if len(training) < MIN_MATCHES:
            print(f"  Spieltag {md:2d}: Übersprungen (nur {len(training)} Trainingsmatches)")
            continue

        # Modell trainieren
        # Referenzdatum = erster Match des Spieltags
        md_matches = [m for m in season_matches if m["matchday"] == md]
        if not md_matches:
            continue
        ref_date = min(m["date"] for m in md_matches)

        model = fit_dixon_coles(training, ref_date)

        # Tipps & Punkte
        md_pts = 0
        md_games = 0
        details = []
        for m in md_matches:
            home, away = m["home"], m["away"]
            if home not in model["attack"] or away not in model["attack"]:
                continue

            th, ta, ev = compute_tip(home, away, model, odds_data or None)

            pts = kicktipp_points(th, ta, m["home_goals"], m["away_goals"])
            md_pts += pts
            md_games += 1
            pts_distribution[pts] += 1
            details.append((home, away, th, ta, m["home_goals"], m["away_goals"], pts))

        avg = md_pts / md_games if md_games else 0
        total_pts += md_pts
        total_games += md_games
        results_by_md.append((md, md_pts, md_games, avg))

        print(f"  Spieltag {md:2d}: {md_pts:3d} Pkt / {md_games} Spiele  (Ø {avg:.2f})")
        if args.verbose:
            for home, away, th, ta, rh, ra, pts in details:
                flag = "✓" if pts >= 2 else "✗"
                print(f"       {flag} {home} vs {away}: Tipp {th}:{ta}, Real {rh}:{ra} → {pts} Pkt")

    if total_games == 0:
        print("Keine Spiele ausgewertet.")
        return

    overall_avg = total_pts / total_games
    max_possible = total_games * 3

    print(f"\n{'='*50}")
    print(f"  Gesamt:     {total_pts} / {max_possible} mögliche Punkte")
    print(f"  Spiele:     {total_games}")
    print(f"  Ø / Spiel:  {overall_avg:.3f} Punkte")
    print(f"  Ø / Spiel (max möglich): 3.000")
    print(f"\n  Punkteverteilung:")
    for p in [3, 2, 1, 0]:
        n = pts_distribution[p]
        pct = n / total_games * 100 if total_games else 0
        bar = "█" * int(pct / 2)
        label = {3: "Exakt  ", 2: "Diff/Remis", 1: "Tendenz   ", 0: "Falsch    "}[p]
        print(f"    {p} Pkt ({label}): {n:3d} ({pct:4.1f}%) {bar}")
    print()


# ---------------------------------------------------------------------------
# Main / Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kicktipp Bundesliga Predictor – Dixon-Coles + Pinnacle-Quoten",
        epilog="Beispiele:\n"
               "  %(prog)s predict --season 2025 --matchday 31\n"
               "  %(prog)s backtest --season 2024 --use-odds\n"
               "  %(prog)s backtest --season 2024 --from-matchday 10 --to-matchday 34 -v\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # predict
    p_pred = sub.add_parser("predict", help="Tipps für einen Spieltag berechnen")
    p_pred.add_argument("--season", type=int, required=True, help="Saison (z.B. 2025 für 2025/26)")
    p_pred.add_argument("--matchday", type=int, required=True, help="Spieltag (1–34)")
    p_pred.add_argument("--use-odds", action="store_true", dest="use_odds",
                        help="Live-Quoten via The Odds API (automatisch wenn ODDS_API_KEY gesetzt)")

    # backtest
    p_back = sub.add_parser("backtest", help="Backtesting über eine Saison")
    p_back.add_argument("--season", type=int, required=True, help="Saison (z.B. 2024 für 2024/25)")
    p_back.add_argument("--from-matchday", type=int, default=1, dest="from_matchday",
                        help="Ab Spieltag (default: 1)")
    p_back.add_argument("--to-matchday", type=int, default=34, dest="to_matchday",
                        help="Bis Spieltag (default: 34)")
    p_back.add_argument("--verbose", "-v", action="store_true",
                        help="Einzelergebnisse ausgeben")
    p_back.add_argument("--use-odds", action="store_true", dest="use_odds",
                        help="Historische Pinnacle-Quoten einbeziehen (football-data.co.uk)")

    # ceiling
    p_ceil = sub.add_parser("ceiling", help="Theoretisches Ceiling analysieren")
    p_ceil.add_argument("--season", type=int, required=True,
                        help="Saison (z.B. 2024 für 2024/25)")
    p_ceil.add_argument("--from-matchday", type=int, default=1, dest="from_matchday")
    p_ceil.add_argument("--to-matchday", type=int, default=34, dest="to_matchday")
    p_ceil.add_argument("--modes", nargs="+",
                        choices=["market", "hindsight", "bins"],
                        default=["market", "hindsight", "bins"],
                        help="Welche Ceilings berechnen (default: alle)")
    p_ceil.add_argument("--compare-model", action="store_true", dest="compare_model",
                        help="Zusätzlich tatsächliche Modell-Performance auf demselben Sample")

    # calibration
    p_cal = sub.add_parser("calibration",
                           help="Reliability-Analyse: Modell-Wahrscheinlichkeiten vs. Realität")
    p_cal.add_argument("--season", type=int, required=True)
    p_cal.add_argument("--from-matchday", type=int, default=1, dest="from_matchday")
    p_cal.add_argument("--to-matchday", type=int, default=34, dest="to_matchday")
    p_cal.add_argument("--use-odds", action="store_true", dest="use_odds",
                       help="Modell + Quoten-Mix testen (sonst nur Dixon-Coles)")

    args = parser.parse_args()

    if args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "ceiling":
        cmd_ceiling(args)
    elif args.cmd == "calibration":
        cmd_calibration(args)


if __name__ == "__main__":
    main()
