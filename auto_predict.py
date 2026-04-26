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

    # Prüfe ob Tipps schon existieren
    TIPS_DIR.mkdir(exist_ok=True)
    tips_file = TIPS_DIR / f"{season}_{season+1}_spieltag_{md:02d}.md"
    if tips_file.exists():
        print(f"Tipps existieren bereits: {tips_file}")
        return

    # Tipps generieren via kicktipp.py
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


def _generate_html(md, season, fixtures, model, live_odds, match_date):
    """Generiert docs/index.html mit den aktuellen Tipps."""
    import kicktipp as kt

    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    match_date_str = match_date.strftime("%d.%m.%Y %H:%M")

    rows = ""
    for f in fixtures:
        home, away = f["home"], f["away"]
        if home not in model["attack"] or away not in model["attack"]:
            rows += f'<tr><td>{home} – {away}</td><td>?</td><td>—</td><td>—</td></tr>\n'
            continue

        th, ta, ev = kt.compute_tip(home, away, model, live_odds or None)
        has_odds = kt._find_odds(live_odds, home, away) is not None
        icon = "&#9889;" if has_odds else ""
        tend = kt.tendency_str(th, ta)
        rows += (f'<tr><td>{icon} {home} – {away}</td>'
                 f'<td><strong>{th}:{ta}</strong></td>'
                 f'<td>{ev:.3f}</td><td>{tend}</td></tr>\n')

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bundesliga Predictor — Spieltag {md}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 700px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 1.6em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover {{ background: #f9f9f9; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
  .legend {{ color: #888; font-size: 0.85em; margin-top: 10px; }}
  footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;
            color: #999; font-size: 0.8em; }}
  a {{ color: #0366d6; text-decoration: none; }}
</style>
</head>
<body>
<h1>Spieltag {md} — Saison {season}/{season+1}</h1>
<p class="meta">
  Erster Anpfiff: {match_date_str} UTC<br>
  Tipps generiert: {now_str} UTC
</p>
<table>
<thead>
<tr><th>Begegnung</th><th>Tipp</th><th>E[Pkt]</th><th>Tendenz</th></tr>
</thead>
<tbody>
{rows}</tbody>
</table>
<p class="legend">&#9889; = mit Pinnacle Live-Quoten</p>
<footer>
  <a href="https://github.com/kontainer-sh/bundesliga-predictor">bundesliga-predictor</a> —
  Dixon-Coles + Pinnacle-Quoten
</footer>
</body>
</html>"""

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML geschrieben: {docs_dir / 'index.html'}")


if __name__ == "__main__":
    main()
