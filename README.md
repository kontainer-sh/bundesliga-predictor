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

#### 1. Dixon-Coles (30% Gewicht)

Das [Dixon-Coles-Modell](https://doi.org/10.1111/1467-9876.00065) (1997) ist die Standardmethode zur Vorhersage von Fußballergebnissen. Jedes Team hat zwei Parameter:

- **Angriffsstärke** α_i — wie viele Tore Team i schießt
- **Abwehrstärke** β_i — wie viele Tore Team i zulässt

Die erwarteten Tore in einem Spiel Home vs. Away sind:

```
λ_home = exp(α_home - β_away + γ)    # γ = Heimvorteil
λ_away = exp(α_away - β_home)
```

Tore folgen einer **Poisson-Verteilung**: P(k Tore) = λ^k · e^(-λ) / k!

Die Wahrscheinlichkeit für ein Ergebnis h:a ist dann P(h) · P(a), korrigiert um einen **Abhängigkeitsfaktor ρ** für niedrige Ergebnisse (0:0, 1:0, 0:1, 1:1), weil diese empirisch häufiger/seltener auftreten als unabhängiges Poisson vorhersagt.

Die Parameter werden per **Maximum-Likelihood-Schätzung** auf historischen Ergebnissen gefittet, mit exponentiellem **Zeitgewicht** (Halbwertszeit 300 Tage), sodass neuere Spiele stärker zählen.

#### 2. Pinnacle-Quoten (70% Gewicht)

Wettquoten des Buchmachers Pinnacle (bekannt für die schärfsten Linien am Markt) werden in eine Score-Matrix umgerechnet:

1. **Overround entfernen**: Die impliziten Wahrscheinlichkeiten P(H), P(D), P(A) werden aus den Dezimalquoten extrahiert und auf 100% normalisiert.
2. **Poisson-Fit**: λ_home und λ_away werden so gewählt, dass die resultierende Poisson-Verteilung die H/D/A-Wahrscheinlichkeiten möglichst gut reproduziert (Minimierung der KL-Divergenz).
3. **Score-Matrix**: P(h:a) für alle Ergebnisse von 0:0 bis 8:8.

Die Score-Matrizen aus Dixon-Coles und Quoten werden **gewichtet gemischt** (70/30), um die Stärken beider Ansätze zu kombinieren.

#### 3. Tipp-Optimierung

Für jeden Kandidaten-Tipp (0:0 bis 2:2) wird der **erwartete Kicktipp-Punkteertrag** berechnet:

```
E[Punkte | Tipp t] = Σ P(h:a) · Punkte(t, h:a)
```

Der Tipp mit dem höchsten Erwartungswert wird gewählt. Dies wird analytisch exakt über die gesamte Score-Matrix berechnet (`numpy.einsum`), nicht per Monte-Carlo-Simulation — mathematisch äquivalent, aber schneller und deterministisch.

#### Warum Poisson?

Die Poisson-Verteilung passt erstaunlich gut auf Fußballtore (empirisch überprüft auf 4.590 Bundesliga-Spielen 2010–2025):

| Kennzahl | Poisson-Annahme | Bundesliga empirisch |
|---|---|---|
| Var/Mean (Heimtore) | 1.000 | 1.154 |
| Var/Mean (Auswärtstore) | 1.000 | 1.120 |
| Häufigstes Ergebnis | 1:1 (10.9%) | 1:1 (11.5%) |

Die leichte Überdispersion (~12–15%) wurde mit Negativer Binomialverteilung getestet, bringt aber keine messbare Verbesserung bei unserem Punkteschema.

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
Market-Ceiling (perfekte Kalibrierung): ~232 Pkt / Saison
Perfektes Oracle:                     810 Pkt / Saison
```

Das Modell erreicht **95% des Market-Ceilings**. Die restlichen 5% sind Kalibrierungsverlust
durch die Poisson-Rekonstruktion aus H/D/A-Quoten — nur mit Correct Score-Quoten behebbar.

In einer 20er-Kicktipp-Liga: Ø Platz 4, ~28% Titelchance, ~95% obere Hälfte.

### Ceiling-Analyse

```bash
python kicktipp.py ceiling --season 2024
python kicktipp.py ceiling --season 2024 --compare-model
python kicktipp.py ceiling --season 2024 --modes market bins
```

## Automatische Tipps (GitHub Action)

Eine GitHub Action generiert täglich um 10:00 MESZ automatisch Tipps für den nächsten Spieltag (wenn dieser innerhalb von 3 Tagen liegt) und legt sie unter `tips/spieltag_XX.md` ab.

Setup: `ODDS_API_KEY` als [GitHub Secret](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions) hinterlegen. Die Action kann auch manuell über den "Run workflow"-Button ausgelöst werden.

## Caching

- **OpenLigaDB / football-data.co.uk**: Dauerhaft in `.cache/` gecacht
- **Live-Quoten**: 6 Stunden Cache (`.cache/live_odds.json`), dann automatisch neu geladen
- Cache löschen für frische Daten: `rm -rf .cache/`

## Punkteschema anpassen

Die vier Konstanten am Anfang von `kicktipp.py` anpassen:

```python
POINTS_TENDENCY = 1       # Sieg: nur Tendenz richtig
POINTS_GOAL_DIFF = 2      # Sieg: Tordifferenz richtig
POINTS_DRAW_TENDENCY = 2  # Unentschieden: Tendenz richtig
POINTS_EXACT = 3          # Exaktes Ergebnis
```

Gängige Kicktipp-Schemata:

| Schema | Tendenz | Differenz | Remis | Exakt |
|---|---|---|---|---|
| 1-2-3 (unser Default) | 1 | 2 | 2 | 3 |
| 2-3-4 (Kicktipp-Standard) | 2 | 3 | 3 | 4 |
| 0-2-3 | 0 | 2 | 2 | 3 |

Nach Änderung des Schemas sollten die Hyperparameter (vor allem `MAX_TIP_GOALS` und `ODDS_WEIGHT`) per Backtest überprüft werden.

## Lizenz

MIT
