#!/usr/bin/env python3
"""Automatische Tippabgabe bei Kicktipp.

Loggt sich ein, holt das Tippabgabe-Formular des nächsten offenen Spieltags,
ordnet die Modell-Tipps (gleiche Pipeline wie auto_predict) den Spielen zu und
gibt sie ab.

SICHERHEIT:
- DRY-RUN ist Default: es wird nur angezeigt, was getippt würde. Erst mit
  --submit wird wirklich abgegeben.
- Credentials NUR aus Umgebungsvariablen, nie im Repo:
    KICKTIPP_EMAIL, KICKTIPP_PASSWORD   (in GitHub Actions als Secrets, wie ODDS_API_KEY)
    KICKTIPP_COMMUNITY  (erforderlich: Runden-Slug; ohne ihn bricht das Skript ab)
- Kann ein Spiel nicht eindeutig einem Modell-Tipp zugeordnet werden, bricht das
  Skript ab und gibt NICHTS ab.
- Bereits getippte Spiele bleiben unangetastet (außer --overwrite).
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, ".")
import kicktipp as kt
from auto_predict import find_next_matchday

BASE = "https://www.kicktipp.de"

# Kicktipp-Anzeigenamen → OpenLigaDB (nur Abweichungen; Rest via _normalize_team)
KT_TO_OLDB = {
    "1899 Hoffenheim": "TSG Hoffenheim",
    "Bor. Mönchengladbach": "Borussia Mönchengladbach",
    "FSV Mainz 05": "1. FSV Mainz 05",
    "Werder Bremen": "SV Werder Bremen",
    "SV Elversberg": "SV 07 Elversberg",
    "Bayern München": "FC Bayern München",
}


def _oldb(name: str) -> str:
    return KT_TO_OLDB.get(name, kt._normalize_team(name))


def login(session: requests.Session, email: str, password: str) -> None:
    r = session.post(f"{BASE}/info/profil/loginaction",
                     data={"kennung": email, "passwort": password}, timeout=30)
    r.raise_for_status()
    # Bei Erfolg leitet Kicktipp weiter; bei Fehler kommt die Login-Seite zurück.
    if "Zugangsdaten sind fehlerhaft" in r.text:
        raise SystemExit("Login fehlgeschlagen — KICKTIPP_EMAIL/PASSWORD prüfen.")


def fetch_form(session: requests.Session, community: str):
    """Holt das Tippabgabe-Formular des nächsten offenen Spieltags.

    Gibt (base_fields, games) zurück:
      base_fields: alle Formularfelder (hidden + submit) als name→value
      games: Liste {tid, home, away, heim_field, gast_field, has_tip}
    """
    r = session.get(f"{BASE}/{community}/tippabgabe", timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form", action=f"/{community}/tippabgabe")
    if form is None:
        raise SystemExit("Tippabgabe-Formular nicht gefunden (eingeloggt? Runde korrekt?).")

    base_fields = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        if name and not re.search(r"\.(heim|gast)Tipp$", name):
            base_fields[name] = inp.get("value", "")

    games = []
    for row in form.select("tbody tr"):
        heim_inp = row.find("input", attrs={"name": re.compile(r"\.heimTipp$")})
        gast_inp = row.find("input", attrs={"name": re.compile(r"\.gastTipp$")})
        if not (heim_inp and gast_inp):
            continue
        tid = re.search(r"spieltippForms\[(\d+)\]", heim_inp["name"]).group(1)
        # Teamzellen: Klasse 'nw', aber nicht die Termin- oder Quoten-Zelle.
        team_cells = [td for td in row.find_all("td")
                      if td.get("class") and "nw" in td["class"]
                      and "kicktipp-time" not in td["class"]
                      and "quoten" not in td["class"]]
        if len(team_cells) < 2:
            continue
        games.append({
            "tid": tid,
            "home": team_cells[0].get_text(strip=True),
            "away": team_cells[1].get_text(strip=True),
            "heim_field": heim_inp["name"],
            "gast_field": gast_inp["name"],
            "has_tip": bool((heim_inp.get("value") or "").strip()
                            or (gast_inp.get("value") or "").strip()),
        })
    if not games:
        raise SystemExit("Keine offenen Spiele im Formular (Deadline vorbei?).")
    return base_fields, games


def compute_model_tips(games):
    """Berechnet Modell-Tipps (Modell + Live-Odds) für die Formular-Spiele."""
    info = find_next_matchday()
    if info is None:
        raise SystemExit("Kein kommender Spieltag (Saisonpause?).")
    season, md = info["season"], info["matchday"]
    all_matches = kt.load_all_matches(season)
    ref_date = datetime.now(tz=timezone.utc)
    model = kt.fit_dixon_coles(
        kt.training_split(all_matches, season, md, ref_date=ref_date), ref_date)
    live_odds = kt.fetch_live_odds() if os.environ.get("ODDS_API_KEY") else {}

    tips, missing = {}, []
    for g in games:
        h, a = _oldb(g["home"]), _oldb(g["away"])
        if h not in model["attack"] or a not in model["attack"]:
            missing.append(f"{g['home']} ({h}) – {g['away']} ({a})")
            continue
        th, ta, _ = kt.compute_tip(h, a, model, live_odds or None)
        tips[g["tid"]] = (th, ta)
    return tips, missing, f"{season}/{season+1} Spieltag {md}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true",
                    help="Tipps WIRKLICH abgeben (sonst nur Dry-Run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="auch bereits getippte Spiele überschreiben")
    ap.add_argument("--community", default=os.environ.get("KICKTIPP_COMMUNITY"),
                    help="Kicktipp-Runde (Slug); Default aus KICKTIPP_COMMUNITY")
    args = ap.parse_args()

    email = os.environ.get("KICKTIPP_EMAIL")
    password = os.environ.get("KICKTIPP_PASSWORD")
    if not (email and password):
        raise SystemExit("KICKTIPP_EMAIL / KICKTIPP_PASSWORD nicht gesetzt.")
    if not args.community:
        raise SystemExit("KICKTIPP_COMMUNITY (Runden-Slug) nicht gesetzt.")

    session = requests.Session()
    session.headers["User-Agent"] = "kicktipp-autotip/1.0"
    login(session, email, password)

    base_fields, games = fetch_form(session, args.community)
    tips, missing, label = compute_model_tips(games)

    if missing:
        print("FEHLER — Spiele ohne Modell-Zuordnung (KT_TO_OLDB prüfen):")
        for m in missing:
            print(f"  {m}")
        raise SystemExit("Abbruch: nichts abgegeben (keine blinden Tipps).")

    print(f"\n{label} — {'ABGABE' if args.submit else 'DRY-RUN'} (Runde '{args.community}')\n")
    to_submit = 0
    data = dict(base_fields)
    for g in games:
        th, ta = tips[g["tid"]]
        skip = g["has_tip"] and not args.overwrite
        mark = "  [schon getippt — übersprungen]" if skip else ""
        print(f"  {g['home']:26} – {g['away']:26}  {th}:{ta}{mark}")
        if skip:
            continue
        to_submit += 1
        data[g["heim_field"]] = str(th)
        data[g["gast_field"]] = str(ta)
        data[f"spieltippForms[{g['tid']}].tippAbgegeben"] = "true"

    if not args.submit:
        print(f"\nDRY-RUN: {to_submit} Spiele würden abgegeben. "
              f"Mit --submit wirklich absenden.")
        return
    if to_submit == 0:
        print("\nNichts abzugeben (alle schon getippt; --overwrite zum Ersetzen).")
        return

    resp = session.post(f"{BASE}/{args.community}/tippabgabe", data=data, timeout=30)
    resp.raise_for_status()
    print(f"\n✓ {to_submit} Tipps abgegeben (HTTP {resp.status_code}).")


if __name__ == "__main__":
    main()
