# Experimente

Protokoll von Methoden, die als Verbesserung gegenüber dem Produktiv-Stack
(Dixon-Coles + Pinnacle-Quoten 70/30, 8×8-Matrix, EV-optimiert) getestet
und entweder etabliert oder verworfen wurden.

---

## 2026-05-10 — Score-Matrix-Recalibration (verworfen)

**Hypothese:** Eine empirisch gelernte multiplikative Korrektur pro Score-Cell
(`actual_freq / predicted_freq` aus Vorsaisons) verbessert die erwarteten
Kicktipp-Punkte. Inspiration: Wilkens 2026 (Bundesliga, +10% ROI mit isotoner
1X2-Recalibration).

**Setup:**
- Test-Saison: 2024/2025 BL1 (306 Spiele, 34 Spieltage)
- Walk-Forward: pro Spieltag DC neu gefittet auf alle Daten davor
- Quoten: football-data.co.uk (Bet365/Pinnacle-Konsens), 70/30 Mix
- Recalibration trainiert auf 2022/2023 + 2023/2024 (kein Data-Leakage)
- `kt.fit_recalibration(min_obs=20)` — nur Cells mit ≥20 historischen Beobachtungen
  bekommen Korrektur, Rest bleibt 1.0

**Ergebnis:**

| | Baseline | Recalibration | Δ |
|---|---|---|---|
| Gesamtpunkte (306 Spiele) | 237 | 228 | **−9 (−3.8%)** |
| Ø Punkte/Spiel | 0.775 | 0.745 | −0.029 |

Gepaarte Statistik (Recal − Baseline pro Spiel):
- Mittelwert: −0.029
- 95%-CI: [−0.146, +0.087]
- Paired Bootstrap p-value (10k): **0.645**

**Fazit:** Kein signifikanter Effekt, leichter Trend ins Negative. Mögliche Gründe:
- Nur 11 von 81 Score-Cells erreichten `min_obs=20` — Korrekturmasse zu dünn
- Pinnacle-Quoten kalibrieren die Score-Matrix bereits sehr nahe am Optimum;
  zusätzliche multiplikative Korrektur introduziert Rauschen
- Wilkens' Erfolg basiert auf isotoner 1X2-Recalibration — eine andere Mathematik
  als die hier getestete per-Cell-Frequenz-Korrektur

**Aktion:** Recalibration NICHT in `auto_predict.py` aktivieren. Der `correction_table=`-
Parameter in `kt.compute_tip` und die Funktionen `fit_recalibration` /
`recalibrate_score_matrix` bleiben im Code als Infrastruktur, falls eine
methodisch andere Recalibration (z.B. echte isotone Regression auf 1X2)
später getestet werden soll.

**Reproduktion:** `python backtest_recalibration.py` (~10 Min, alle Daten gecached).

---

## 2026-05-10 — λ-Sweep Modell vs. Quoten (verworfen)

**Hypothese:** Das produktive Mischgewicht `ODDS_WEIGHT = 0.7` ist nicht
zwingend optimal. Ein Sweep über λ ∈ {0.0, 0.1, ..., 1.0} sollte das
empirisch beste Mischverhältnis finden (oder bestätigen).

**Setup:**
- Test-Saisons: 2022/2023, 2023/2024, 2024/2025 BL1 (zusammen 918 Spiele)
- Walk-Forward pro Spieltag, DC einmal pro Spieltag gefittet, Tipps für
  alle 11 λ-Werte aus derselben Score-Matrix berechnet
- Quoten: football-data.co.uk (Bet365/Pinnacle-Konsens)

**Aggregat-Ergebnis:**

| λ | Σ Pkt | Δ vs 0.7 |
|---|---|---|
| 0.0 (nur Modell) | 719 | −18 |
| 0.3 | **752** | **+15** ← Aggregat-Maximum |
| 0.5 | 741 | +4 |
| 0.7 (Produktion) | 737 | 0 |
| 0.9 | 734 | −3 |
| 1.0 (nur Quoten) | 742 | +5 |

Gepaarte Diff (λ=0.3 − λ=0.7): +0.016 Pkt/Spiel, 95%-CI [−0.010, +0.044],
Bootstrap p=0.239 (n=918).

**Pro-Saison-Optima (Robustheits-Check):**

| Saison | bestes λ |
|---|---|
| 2022/2023 | 0.3 |
| 2023/2024 | 1.0 |
| 2024/2025 | 0.9 |

**Fazit:** Optima driften zwischen den Saisons über die volle Range —
das Aggregat-Maximum bei 0.3 wird im Wesentlichen von 2022/2023 getragen.
Statistisch nicht von 0.7 unterscheidbar (p=0.24). Es gibt kein robustes
empirisches Optimum.

**Zwei nutzbare Erkenntnisse trotz Null-Ergebnis:**

1. **Quoten allein (719 → 742) sind ~23 Pkt besser als das Modell allein.**
   Pinnacle-Konsens dominiert — wie aus der Literatur erwartet.
2. **Die beste Mischung (752) schlägt Quoten allein (742) um 10 Pkt.**
   Das DC-Modell trägt einen kleinen, aber realen Mehrwert über reine
   Markt-Replikation hinaus. → Modell-Verbesserungen (GAS, pi-Rating)
   sind nicht verschwendet, aber das absolute Hebelpotenzial ist klein
   (Größenordnung ~10-30 Pkt/Saison).

**Aktion:** `ODDS_WEIGHT = 0.7` bleibt. Datenlage rechtfertigt keine
Änderung. Falls eine zukünftige Modell-Verbesserung den Modell-Beitrag
deutlich erhöht, sollte der λ-Sweep wiederholt werden.

**Reproduktion:** `python backtest_lambda_sweep.py` (~10 Min, gecached).

---

## Backlog (aus Paper-Recherche, ungetestet)

Methoden, die in der Literatur DC schlagen und mit unseren Daten machbar wären:

1. **Isotone 1X2-Recalibration** (Wilkens 2026) — andere Mathematik als die
   verworfene per-Cell-Variante. Trainiert separate isotone Regressionen für
   P(home), P(draw), P(away) gegen empirische Häufigkeiten. Der nächste sinnvolle
   Recal-Versuch.
2. **Score-driven Team-Stärken (GAS)** (Koopman & Lit 2015/2019) — dynamische
   Updates aus Likelihood-Score statt fixer 300-Tage-Halbwertszeit. Geschätzter
   Aufwand 2-4 Tage.
3. **pi-Rating** (Constantinou 2013) — Home/Away-Ratings, Update über Tor-
   Differenz. Sehr leichtgewichtig (~50 Zeilen). Als alternativer Modell-Layer
   neben DC denkbar.
4. **Bayesianische λ-Schätzung** (Egidi/Pauli/Torelli 2018) — ersetzt fixes 70/30
   durch gelernte konvexe Kombination. Methodisch eleganter, praktischer
   Mehrwert vermutlich klein.
