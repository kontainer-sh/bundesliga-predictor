# Kicktipp Bundesliga Predictor

Statistisches Vorhersagemodell für die Bundesliga, optimiert auf das Kicktipp-Punkteschema.

Kombiniert ein **Dixon-Coles-Modell** (Teamstärke aus historischen Ergebnissen) mit **Pinnacle-Wettquoten** (Markt-Konsens) zu einem erwartungswert-optimalen Tipp.

## Ergebnisse (Backtest Saison 2024/25)

| Modus | Punkte (30 ST) | Ø / Spiel |
|---|---|---|
| Nur Modell | 211 | 0.781 |
| Modell + Quoten | **231** | **0.856** |

Punkteschema: Tendenz 1, Tordifferenz/Remis 2, Exakt 3 (konfigurierbar im Code).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: API-Key für Live-Quoten ([The Odds API](https://the-odds-api.com), kostenlos, 500 Credits/Monat):

```bash
echo "ODDS_API_KEY=dein_key" > .env
```

## Verwendung

### Tipps für den nächsten Spieltag

```bash
python kicktipp.py predict --season 2025 --matchday 31
```

Mit `ODDS_API_KEY` in `.env` werden automatisch aktuelle Pinnacle-Quoten einbezogen (markiert mit ⚡). Ohne Key wird nur das Dixon-Coles-Modell genutzt.

### Backtesting

```bash
# Nur Modell
python kicktipp.py backtest --season 2024

# Mit historischen Quoten
python kicktipp.py backtest --season 2024 --use-odds

# Einzelergebnisse anzeigen
python kicktipp.py backtest --season 2024 --use-odds -v

# Bestimmter Spieltag-Bereich
python kicktipp.py backtest --season 2024 --from-matchday 10 --to-matchday 30
```

## Wie es funktioniert

### Datenquellen

| Quelle | Daten | Kosten |
|---|---|---|
| [OpenLigaDB](https://www.openligadb.de) | Ergebnisse 1. + 2. Bundesliga | Kostenlos |
| [football-data.co.uk](https://www.football-data.co.uk) | Historische Pinnacle-Quoten (Backtest) | Kostenlos |
| [The Odds API](https://the-odds-api.com) | Live-Quoten vor Spieltag (Predict) | Kostenlos (500 Req/Monat) |

### Modell

1. **Dixon-Coles** (30% Gewicht): Schätzt Angriffs-/Abwehrstärke jedes Teams aus historischen Ergebnissen (3 Vorsaisons + 2. Liga, Poisson-Verteilung mit Zeitgewichtung).
2. **Pinnacle-Quoten** (70% Gewicht): Markt-implizierte Wahrscheinlichkeiten (H/D/A), umgewandelt in Score-Matrix via Poisson-Fit.
3. **Tipp-Optimierung**: Wählt den Tipp mit dem höchsten erwarteten Punkteertrag unter dem konfigurierten Kicktipp-Punkteschema. Analytisch exakt berechnet (kein Monte Carlo nötig).

### Optimierte Hyperparameter (Cross-Validation über 3 Saisons)

| Parameter | Wert | Bedeutung |
|---|---|---|
| `HALF_LIFE_DAYS` | 300 | Zeitgewichtung: ~10 Monate Halbwertszeit |
| `MAX_TIP_GOALS` | 2 | Tipps nur bis 2:2 (konservativ) |
| `NUM_PREV_SEASONS` | 3 | Trainingsdaten aus 3 Vorsaisons |
| `ODDS_WEIGHT` | 0.7 | 70% Quoten, 30% Modell |

### Was wir getestet haben (und was nicht hilft)

| Ansatz | Ergebnis |
|---|---|
| xG (Expected Goals) | ±0 — schon in den Quoten enthalten |
| Negative Binomialverteilung | ±0 — Poisson passt gut genug |
| L2-Regularisierung | ±0 — genug Trainingsdaten |
| Over/Under-Quoten | -11 Pkt — verschlechtert Tendenz-Trefferquote |
| Teamspezifischer Heimvorteil | -13 Pkt — Overfitting |
| Remis-Boost | -13 Pkt — Modell optimiert schon korrekt |

## Theoretische Grenzen

```
Immer 2:1 tippen (uninformiert):     ~192 Pkt / Saison
Unser Modell (mit Quoten):           ~231 Pkt / Saison
Realistisches Ceiling:               ~245 Pkt / Saison
Perfektes Oracle:                     810 Pkt / Saison
```

In einer 20er-Kicktipp-Liga: Ø Platz 4, ~28% Titelchance, ~95% obere Hälfte.

## Caching

- **OpenLigaDB / football-data.co.uk**: Dauerhaft in `.cache/` gecacht
- **Live-Quoten**: 6 Stunden Cache (`.cache/live_odds.json`), dann automatisch neu geladen
- Cache löschen für frische Daten: `rm -rf .cache/`

## Punkteschema anpassen

Das Punkteschema ist in der Funktion `kicktipp_points()` definiert. Aktuell:

- **Sieg**: Tendenz 1, Tordifferenz 2, Exakt 3
- **Unentschieden**: Tendenz 2, Exakt 3

Für andere Kicktipp-Regeln einfach die Funktion anpassen und Hyperparameter neu tunen.

## Lizenz

MIT
