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
            rows += f'<tr><td class="match">{home} – {away}</td><td class="tip">?</td><td class="ev">—</td><td>—</td></tr>\n'
            continue

        th, ta, ev = kt.compute_tip(home, away, model, live_odds or None)
        has_odds = kt._find_odds(live_odds, home, away) is not None
        badge = '<span class="odds-badge">&#9889; Odds</span>' if has_odds else ""
        tend = kt.tendency_str(th, ta)
        tend_class = {"Heimsieg": "tend-home", "Auswärtssieg": "tend-away", "Unentschieden": "tend-draw"}[tend]
        rows += (f'<tr><td class="match">{home} – {away}{badge}</td>'
                 f'<td class="tip">{th}:{ta}</td>'
                 f'<td class="ev">{ev:.3f}</td>'
                 f'<td class="tend {tend_class}">{tend}</td></tr>\n')

    season_str = f"{season}/{season+1}"
    title = f"Bundesliga Tipps Spieltag {md} — Saison {season_str}"
    desc = (f"Statistisch optimierte Bundesliga-Tipps für Spieltag {md} "
            f"(Saison {season_str}). Dixon-Coles-Modell kombiniert mit "
            f"Pinnacle-Wettquoten. Aktualisiert vor jedem Spieltag.")
    canonical = "https://kontainer.sh/bundesliga-predictor/"

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="robots" content="index, follow">
<link rel="canonical" href="{canonical}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:url" content="{canonical}">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 760px; margin: 0 auto; padding: 24px 20px; color: #1a1a1a;
         background: #fafafa; }}
  header {{ margin-bottom: 32px; }}
  h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
  h1 span {{ color: #888; font-weight: 400; font-size: 0.75em; }}
  .subtitle {{ color: #555; font-size: 1.05em; margin-bottom: 16px; }}
  .meta {{ color: #888; font-size: 0.85em; line-height: 1.6; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
           padding: 20px; margin: 24px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 10px 12px; color: #666; font-weight: 600;
       font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.03em;
       border-bottom: 2px solid #e5e5e5; }}
  td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8f9fa; }}
  .match {{ font-weight: 500; }}
  .tip {{ font-size: 1.15em; font-weight: 700; color: #1a1a1a; text-align: center; }}
  .ev {{ text-align: center; color: #666; font-size: 0.9em; }}
  .tend {{ font-size: 0.9em; }}
  .tend-home {{ color: #2563eb; }}
  .tend-draw {{ color: #7c3aed; }}
  .tend-away {{ color: #dc2626; }}
  .odds-badge {{ display: inline-block; background: #fef3c7; color: #92400e;
                 font-size: 0.7em; padding: 2px 6px; border-radius: 3px;
                 margin-left: 6px; vertical-align: middle; }}
  .legend {{ color: #999; font-size: 0.8em; margin-top: 8px; padding: 0 12px; }}
  .method {{ margin: 24px 0; }}
  .method summary {{ cursor: pointer; color: #555; font-size: 0.95em; font-weight: 500; }}
  .method-content {{ margin-top: 12px; color: #666; font-size: 0.88em; line-height: 1.7; }}
  .method-content p {{ margin-bottom: 8px; }}
  footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e5e5;
            color: #aaa; font-size: 0.8em; line-height: 1.8; }}
  footer a {{ color: #666; text-decoration: none; }}
  footer a:hover {{ text-decoration: underline; }}
  @media (max-width: 500px) {{
    .match {{ font-size: 0.9em; }}
    th, td {{ padding: 8px 6px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Bundesliga Tipps <span>Spieltag {md}</span></h1>
  <p class="subtitle">Saison {season_str}</p>
  <p class="meta">
    Anpfiff: {match_date_str} UTC &middot;
    Aktualisiert: {now_str} UTC
  </p>
</header>

<div class="card">
<table>
<thead>
<tr><th>Begegnung</th><th style="text-align:center">Tipp</th><th style="text-align:center">E[Pkt]</th><th>Tendenz</th></tr>
</thead>
<tbody>
{rows}</tbody>
</table>
<p class="legend">&#9889; mit Pinnacle-Quoten &middot; ohne Badge = nur Modell</p>
</div>

<details class="method">
<summary>Punkteschema</summary>
<div class="method-content">
<p>Optimiert auf folgendes Kicktipp-Punkteschema:</p>
<table style="max-width:400px; margin: 8px 0 12px 0;">
<tr><td><strong>Exaktes Ergebnis</strong></td><td style="text-align:right"><strong>{kt.POINTS_EXACT} Punkte</strong></td></tr>
<tr><td>Richtige Tordifferenz (Sieg)</td><td style="text-align:right">{kt.POINTS_GOAL_DIFF} Punkte</td></tr>
<tr><td>Richtige Tendenz (Unentschieden)</td><td style="text-align:right">{kt.POINTS_DRAW_TENDENCY} Punkte</td></tr>
<tr><td>Richtige Tendenz (Sieg)</td><td style="text-align:right">{kt.POINTS_TENDENCY} Punkt</td></tr>
<tr><td>Falsch</td><td style="text-align:right">0 Punkte</td></tr>
</table>
<p>E[Pkt] = erwarteter Punkteertrag des Tipps unter diesem Schema.</p>
</div>
</details>

<details class="method">
<summary>Wie werden die Tipps berechnet?</summary>
<div class="method-content">
<p>Die Tipps kombinieren zwei Ansätze: Ein <strong>Dixon-Coles-Modell</strong> (30%)
schätzt die Angriffs- und Abwehrstärke jedes Teams aus historischen Ergebnissen
der letzten 3+ Saisons (1. und 2. Bundesliga). <strong>Pinnacle-Wettquoten</strong> (70%)
liefern den Markt-Konsens aus tausenden informierter Wetter.</p>
<p>Beide Quellen werden zu einer Wahrscheinlichkeitsmatrix für alle möglichen
Ergebnisse kombiniert. Der Tipp mit dem höchsten <strong>erwarteten Punkteertrag</strong>
wird gewählt — analytisch exakt berechnet.</p>
<p>Im Backtest erreicht das Modell <strong>95% des theoretischen Maximums</strong>.
In einer 20er-Liga: durchschnittlich Platz 4, ~28% Titelchance.</p>
</div>
</details>

<footer>
  <a href="https://github.com/kontainer-sh/bundesliga-predictor">Quellcode auf GitHub</a>
  &middot; Open Source (MIT)
  &middot; Dixon-Coles + Pinnacle-Quoten<br>
  Daten: <a href="https://www.openligadb.de">OpenLigaDB</a>,
  <a href="https://www.football-data.co.uk">football-data.co.uk</a>,
  <a href="https://the-odds-api.com">The Odds API</a>
</footer>
</body>
</html>"""

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML geschrieben: {docs_dir / 'index.html'}")


if __name__ == "__main__":
    main()
