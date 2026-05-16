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

## 2026-05-16 — Disagreement-Test Modell vs. reine Quoten (Null-Resultat)

**Hypothese:** Wenn das DC-Modell echten Edge gegenüber reiner Markt-Replikation
(λ=1.0, nur Pinnacle-Quoten) hat, sollte sich dieser auf den Spielen
*manifestieren, bei denen beide Strategien unterschiedlich tippen*. Agreement-
Spiele liefern null diagnostische Information.

**Setup:**
- 3 Saisons (2022/23, 2023/24, 2024/25), 918 BL1-Spiele
- Walk-Forward: DC pro Spieltag neu gefittet
- Zwei Strategien getestet: λ=0.7 (Produktion) und λ=0.3 (Aggregat-Optimum
  aus λ-Sweep) gegen λ=1.0 (nur Quoten)
- Paired Bootstrap (n=10.000) auf Disagreement-Spielen

**Ergebnis (λ=0.7, Produktion, vs λ=1.0):**

| | Modell λ=0.7 | Quoten λ=1.0 | Δ |
|---|---|---|---|
| Alle 918 Spiele | 737 | 742 | −5 |
| Disagreement (142, 15.5%) | 101 | 106 | −5 |

Paired Bootstrap: Δ = −0.035 Pkt/Spiel, 95%-CI [−0.20, +0.13], **p = 0.72**.
Per-Saison: 2022/23 +12, 2023/24 −4, 2024/25 −13 (Vorzeichen dreht).
Tendenz-Disagreements (42 Spiele): −10 Pkt (Modell überstimmt Markt-Tendenz und verliert).

**Ergebnis (λ=0.3, Aggregat-Optimum, vs λ=1.0):**

| | Modell λ=0.3 | Quoten λ=1.0 | Δ |
|---|---|---|---|
| Alle 918 Spiele | 752 | 742 | +10 |
| Disagreement (263, 28.6%) | 204 | 194 | +10 |

Paired Bootstrap: Δ = +0.038 Pkt/Spiel, 95%-CI [−0.09, +0.17], **p = 0.59**.
Per-Saison: 2022/23 **+22**, 2023/24 −3, 2024/25 −9 — der gesamte aggregierte
+10-Vorteil stammt aus *einer* Saison.

**Fazit:** Weder die Produktionseinstellung (λ=0.7) noch das scheinbare λ-Sweep-
Optimum (λ=0.3) zeigen statistisch signifikanten Edge gegenüber reiner Markt-
Replikation. Der im λ-Sweep beobachtete +10-Pkt-Vorteil von λ=0.3 ist ein
Single-Season-Artefakt aus 2022/23 und in den letzten zwei Saisons sogar
ins Negative gedreht.

Das bestätigt unabhängig die Schlussfolgerung des λ-Sweeps („Optima driften
über die volle Range") mit der schärferen Disagreement-Metrik. Praktisch
heißt das: das DC-Modell trägt für unsere Daten und Kicktipp-Punkteschema
keinen empirisch nachweisbaren Mehrwert über reine Pinnacle-Closing-Odds
hinaus.

**Aktion:** `ODDS_WEIGHT = 0.7` bleibt — eine Änderung auf 0.3 oder 1.0 ist
nicht datengestützt. Modell-Verbesserungen (GAS, pi-Rating, isotone
Recalibration) müssten den DC-Beitrag substantiell stärken, bevor sich ein
λ-Sweep neu lohnt.

**Reproduktion:** `python backtest_disagreement.py [lambda]` (~1.3 Min, gecached).

---

## 2026-05-16 — EV-Gap-Sensitivität (diagnostisch, kein Edge messbar)

**Hypothese:** Falls das DC-Modell überhaupt einen Hebel gegenüber reinen
Quoten hat, sollte er sich in Spielen mit kleinem EV-Gap konzentrieren —
also dort, wo `EV(bester Tipp) - EV(zweitbester Tipp)` klein ist und die
Tippentscheidung knapp wird. Bei großem Gap stimmen Modell und Quoten ohnehin
fast immer überein.

**Setup:**
- 3 Saisons (2022/23, 2023/24, 2024/25), 918 BL1-Spiele
- Walk-Forward: DC pro Spieltag neu gefittet
- Pro Spiel: EV-Gap der Produktions-Score-Matrix (λ=0.7) berechnet, in 5 Bins eingeteilt
- Pro Bin: Modell-Punkte vs. Quoten-Punkte (λ=1.0) verglichen, Paired Bootstrap (n=10.000)

**EV-Gap-Verteilung:** min=0.000, median=0.013, max=0.087 Pkt/Spiel — alle
Entscheidungen sind sehr knapp; das Modell sieht selten einen klar
überlegenen Tipp.

**Ergebnis pro Bin:**

| Bin (EV-Gap) | N | Disagree | Modell | Quoten | Δ | p |
|---|---|---|---|---|---|---|
| [0.000, 0.005) | 189 | 83 (44%) | 146 | 138 | +8 | 0.30 |
| [0.005, 0.010) | 194 | 43 (22%) | 137 | 143 | −6 | 0.41 |
| [0.010, 0.020) | 243 | 11 (4.5%) | 194 | 193 | +1 | 0.92 |
| [0.020, 0.040) | 249 | 5 (2%) | 208 | 216 | −8 | **0.032** |
| [0.040, 0.100) | 43 | 0 | 52 | 52 | 0 | — |

**Disagreement-Konzentration:** 88.7% aller 142 Disagreement-Spiele liegen in
den zwei niedrigsten Bins (Gap < 0.010). Bei Gap ≥ 0.040 stimmen Modell und
Quoten zu 100% überein.

**Fazit:**

1. Kein systematischer Edge über die Bins — Vorzeichen oszilliert.
2. Einziger p<0.05-Befund (Bin [0.020, 0.040): Modell *verliert* signifikant,
   p=0.032). Mit Bonferroni-Korrektur (5 Bins) p_adj=0.16 → nicht mehr
   signifikant. Multiple-Testing entwertet den Einzelbefund.
3. Das DC-Modell hat strukturell nur in ~14% der Spiele (Gap < 0.010)
   überhaupt einen Hebel. Best-Case-Schätzung für eine perfekte DC-Verbesserung:
   ~20–30 Saisonpunkte. Größenordnung deckt sich mit λ-Sweep-Befund.

Zusammen mit Disagreement-Test (gleicher Datumseintrag): Der DC-Beitrag ist
weder im Aggregat noch in irgendeinem EV-Gap-Bin von Rauschen unterscheidbar.

**DC bleibt im Code — aus Robustheits-Gründen, nicht wegen Edge:**

Der Fallback-Pfad in `kicktipp.py:822-830` verwendet automatisch reines DC,
wenn für ein Spiel keine Quoten vorliegen (Odds-API-Ausfall, fehlende
Liga-Abdeckung, etc.). Diese Funktion ist unabhängig von `ODDS_WEIGHT`:

```python
od = _find_odds(odds_dict, home, away) if odds_dict else None
if od:
    combined = (1 - ODDS_WEIGHT) * dc_mat + ODDS_WEIGHT * odds_mat
else:
    combined = score_matrix(home, away, model)   # ← reines DC als Fallback
```

Damit ist die Frage „statistischer Edge des Mix" entkoppelt von „brauchen wir
das DC-Modell". DC bleibt als Infrastruktur erhalten; die einzige offene
Frage ist `ODDS_WEIGHT` selbst, und der bleibt mangels Datengrundlage bei 0.7.

**Aktion:** Keine Code-Änderung. Future-DC-Verbesserungen (GAS, pi-Rating,
isotone Recal) müssten den DC-Beitrag substantiell stärken, um den
Best-Case-Hebel von ~20–30 Pkt/Saison auszuschöpfen.

**Reproduktion:** `python backtest_ev_gap.py` (~1.3 Min, gecached).

---

## 2026-05-16 — Calibration-Test der Score-Matrix (diagnostisch)

**Hypothese / Frage:** Sind die aggregierten Markt-Wahrscheinlichkeiten (1X2,
Over 2.5, BTTS), die sich aus der 8×8-Score-Matrix ergeben, probabilistisch
gut kalibriert? Wenn die Matrix systematisch fehl-kalibriert wäre, ließe
sich daraus ein Hebel ableiten — z.B. via gezielter Recalibration.

**Setup:**
- 3 Saisons (2022/23, 2023/24, 2024/25), 918 BL1-Spiele mit Quoten
- Walk-Forward, drei Strategien parallel ausgewertet:
  λ=0.0 (DC pur), λ=0.7 (Produktion), λ=1.0 (Quoten pur)
- Metriken: Brier, LogLoss, Expected Calibration Error (10 Bins), Reliability-Tabelle

**Ergebnis 1X2 (Multi-Class):**

| Strategie | Brier | LogLoss | ECE-H | ECE-D | ECE-A |
|---|---|---|---|---|---|
| DC pur (λ=0.0) | 0.5957 | 0.9970 | **0.0176** | **0.0238** | 0.0214 |
| Mix Prod (λ=0.7) | 0.5862 | 0.9843 | 0.0226 | 0.0306 | 0.0222 |
| Quoten pur (λ=1.0) | **0.5850** | **0.9828** | 0.0247 | 0.0342 | 0.0219 |

**Ergebnis binäre Märkte:**

| Strategie | Markt | Brier | ECE | ⟨p⟩ | ⟨y⟩ | Δ |
|---|---|---|---|---|---|---|
| DC pur | Over 2.5 | 0.2333 | 0.0331 | 0.6048 | 0.6089 | **−0.004** |
| Mix Prod | Over 2.5 | 0.2304 | 0.0470 | 0.5889 | 0.6089 | −0.020 |
| Quoten pur | Over 2.5 | 0.2304 | 0.0429 | 0.5821 | 0.6089 | −0.027 |
| DC pur | BTTS | 0.2382 | 0.0079 | 0.5915 | 0.5926 | **−0.001** |
| Mix Prod | BTTS | 0.2363 | 0.0324 | 0.5712 | 0.5926 | −0.021 |
| Quoten pur | BTTS | 0.2367 | 0.0356 | 0.5625 | 0.5926 | −0.030 |

**Reliability P(Home Win) bei λ=0.7** (gekürzt):
- Bin [0.3, 0.4) (N=215): ⟨p⟩=0.356 vs ⟨y⟩=0.308 → −4.8 pp (Heimsiege werden im mittleren Bereich leicht überschätzt)
- Andere Bins mit N ≥ 50: Abweichungen ±3 pp.

**Drei Beobachtungen:**

1. **Brier vs. ECE divergieren.** Quoten haben besten Brier/LogLoss (höhere
   Auflösung/Schärfe), aber höchste ECE für 1X2 (leicht miskalibriert,
   insbesondere Draws). DC ist besser kalibriert, aber unscharf. Der Mix
   sitzt dazwischen.
2. **Systematische Unterschätzung von Over 2.5 und BTTS um 2–3 Prozentpunkte**
   in beiden quoten-basierten Strategien — bei DC pur fast Null. Konsistent
   mit Buchmacher-Konservativität bei Over-Quoten.
3. **ECE-Werte sind klein (<0.04).** Perfekte Recalibration würde Brier
   um maximal ~0.005 verbessern.

**Fazit:** Die Score-Matrix ist bereits gut kalibriert. Die kleine
Draw-Miskalibration der Quoten (ECE-D=0.034) und der Over/BTTS-Bias (−2 bis
−3 pp) sind real, aber zu klein, um nach Kicktipp-Argmax-Reduktion einen
nutzbaren Hebel zu bieten. Das erklärt post-hoc, warum die Recalibration-
Experimente (per-Cell, 2026-05-10) nicht funktioniert haben: nicht die
Methode war falsch, der Spielraum ist schlicht zu klein.

Konzeptionell rechtfertigt der Test die Mix-Architektur:  DC liefert gute
*Kalibrierung*, Quoten liefern *Schärfe*. Der 70/30-Mix halbiert die ECE-
Lücke zwischen den beiden. Im Argmax-Regime ist dieser Vorteil aber
unsichtbar (siehe Disagreement- und EV-Gap-Tests).

**Aktion:** Keine Code-Änderung. Isotone 1X2-Recalibration (offener
Backlog-Punkt 1) bleibt theoretisch interessant, aber das obere Limit
des erreichbaren Effekts ist nach diesem Test stark eingegrenzt — ECE-D=0.034
über 918 Spiele ist die ganze Munition.

**Reproduktion:** `python backtest_calibration.py` (~1.3 Min, gecached).

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
