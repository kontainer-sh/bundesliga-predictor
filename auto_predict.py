#!/usr/bin/env python3
"""
Automatische Spieltag-Erkennung und Tipp-Generierung.
Wird von der GitHub Action täglich aufgerufen.

- Erkennt den nächsten ungespielen Spieltag via OpenLigaDB
- Generiert Tipps nur wenn der Spieltag in den nächsten 3 Tagen beginnt
- Schreibt Ergebnis nach tips/spieltag_XX.md
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

OPENLIGADB = "https://api.openligadb.de"
TIPS_DIR = Path(__file__).parent / "tips"


def find_next_matchday() -> dict | None:
    """Findet den nächsten ungespielen Spieltag."""
    resp = requests.get(f"{OPENLIGADB}/getmatchdata/bl1", timeout=30)
    resp.raise_for_status()
    matches = resp.json()

    now = datetime.now(tz=timezone.utc)
    future = []
    for m in matches:
        dt_str = m.get("matchDateTimeUTC", "")
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            continue
        results = m.get("matchResults", [])
        final = next((r for r in results if r["resultTypeID"] == 2), None)
        if final is None and dt > now:
            future.append({
                "date": dt,
                "matchday": m["group"]["groupOrderID"],
            })

    if not future:
        return None

    future.sort(key=lambda x: x["date"])
    next_match = future[0]
    days_until = (next_match["date"] - now).total_seconds() / 86400
    season = next_match["date"].year if next_match["date"].month >= 7 else next_match["date"].year - 1

    return {
        "matchday": next_match["matchday"],
        "season": season,
        "date": next_match["date"],
        "days_until": days_until,
    }


def main():
    info = find_next_matchday()

    if info is None:
        print("Kein kommender Spieltag gefunden (Saisonpause?).")
        return

    md = info["matchday"]
    season = info["season"]
    days = info["days_until"]
    date_str = info["date"].strftime("%d.%m.%Y %H:%M")

    print(f"Nächster Spieltag: {md} (Saison {season}/{season+1})")
    print(f"Erster Anpfiff: {date_str} UTC ({days:.1f} Tage)")

    # Nur generieren wenn Spieltag in den nächsten 3 Tagen
    if days > 3:
        print(f"Noch {days:.1f} Tage — zu früh für Tipps.")
        return

    # Tipps werden bei jedem Lauf neu generiert — Quoten verbessern sich
    # näher am Anpfiff, daher ist eine Aktualisierung wertvoll.
    TIPS_DIR.mkdir(exist_ok=True)
    tips_file = TIPS_DIR / f"{season}_{season+1}_spieltag_{md:02d}.md"

    print(f"\nGeneriere Tipps für Spieltag {md}...\n")
    import kicktipp as kt

    all_matches = kt.load_all_matches(season)
    training = [m for m in all_matches
                if not (m["matchday"] >= md and m["date"].year >= season)]

    if len(training) < kt.MIN_MATCHES:
        print(f"Zu wenig Trainingsdaten ({len(training)}).")
        return

    model = kt.fit_dixon_coles(training, datetime.now(tz=timezone.utc))

    live_odds = {}
    if os.environ.get("ODDS_API_KEY"):
        live_odds = kt.fetch_live_odds()

    fixtures = kt.fetch_matchday_fixtures(season, md)

    # Markdown generieren
    lines = [
        f"# Spieltag {md} — Saison {season}/{season+1}",
        f"",
        f"Generiert am {datetime.now().strftime('%d.%m.%Y %H:%M')} UTC",
        f"",
        f"| Begegnung | Tipp | E[Pkt] | Tendenz | Quelle |",
        f"|---|---|---|---|---|",
    ]

    for f in fixtures:
        home, away = f["home"], f["away"]
        if home not in model["attack"] or away not in model["attack"]:
            lines.append(f"| {home} – {away} | ? | — | Team unbekannt | — |")
            continue

        th, ta, ev = kt.compute_tip(home, away, model, live_odds or None)
        has_odds = kt._find_odds(live_odds, home, away) is not None
        source = "Modell + Odds" if has_odds else "Nur Modell"
        tend = kt.tendency_str(th, ta)
        lines.append(f"| {home} – {away} | **{th}:{ta}** | {ev:.3f} | {tend} | {source} |")

    lines.append("")
    tips_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tipps geschrieben: {tips_file}")

    # HTML für GitHub Pages generieren
    _generate_html(md, season, fixtures, model, live_odds, info["date"])

    # Auch für stdout
    print()
    for line in lines:
        print(line)


WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
TEND_CLASS = {"Heimsieg": "tend-home", "Auswärtssieg": "tend-away", "Unentschieden": "tend-draw"}


def _build_rows(fixtures, model, live_odds):
    """Bereitet Zeilendaten + Tagestrenner für das Template auf."""
    from datetime import timedelta
    import kicktipp as kt

    rows = []
    total_ev = 0.0
    last_date = None
    for f in fixtures:
        home, away = f["home"], f["away"]
        kickoff = f.get("kickoff")

        if kickoff:
            kickoff_local = kickoff + timedelta(hours=2)  # MESZ = UTC+2
            ko_str = f'{WOCHENTAGE[kickoff_local.weekday()]} {kickoff_local.strftime("%d.%m. %H:%M")}'
            date_key = kickoff_local.date()
        else:
            ko_str = ""
            date_key = None

        show_separator = bool(date_key and last_date and date_key != last_date)
        last_date = date_key or last_date

        row = {"kickoff_str": ko_str, "home": home, "away": away,
               "show_separator": show_separator, "has_odds": False,
               "tip": None, "ev": None, "tend": None, "tend_class": None}

        if home in model["attack"] and away in model["attack"]:
            th, ta, ev = kt.compute_tip(home, away, model, live_odds or None)
            total_ev += ev
            tend = kt.tendency_str(th, ta)
            row.update(tip=f"{th}:{ta}", ev=ev, tend=tend,
                       tend_class=TEND_CLASS[tend],
                       has_odds=kt._find_odds(live_odds, home, away) is not None)
        rows.append(row)

    return rows, total_ev


def _generate_html(md, season, fixtures, model, live_odds, match_date):
    """Rendert docs/index.html aus templates/index.html.j2."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import kicktipp as kt

    rows, total_ev = _build_rows(fixtures, model, live_odds)
    season_str = f"{season}/{season+1}"

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=False, lstrip_blocks=False,
    )
    html = env.get_template("index.html.j2").render(
        title=f"Bundesliga Tipps Spieltag {md} — Saison {season_str}",
        desc=(f"Statistisch optimierte Bundesliga-Tipps für Spieltag {md} "
              f"(Saison {season_str}). Dixon-Coles-Modell kombiniert mit "
              f"Pinnacle-Wettquoten. Aktualisiert vor jedem Spieltag."),
        canonical="https://kontainer.sh/bundesliga-predictor/",
        md=md, season_str=season_str,
        match_date_str=match_date.strftime("%d.%m.%Y %H:%M"),
        now_str=datetime.now().strftime("%d.%m.%Y %H:%M"),
        rows=rows, total_ev=total_ev,
        points={"exact": kt.POINTS_EXACT, "goal_diff": kt.POINTS_GOAL_DIFF,
                "draw_tendency": kt.POINTS_DRAW_TENDENCY, "tendency": kt.POINTS_TENDENCY},
    )

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML geschrieben: {docs_dir / 'index.html'}")


if __name__ == "__main__":
    main()
