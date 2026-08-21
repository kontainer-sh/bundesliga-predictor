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

## 2026-07-14 — Literatur-Review: Schlägt irgendetwas Pinnacle-Closing-Odds? (Deep Research)

**Frage:** Gibt es publizierte Evidenz (2020–2026), dass ein öffentliches Modell
oder eine Datenquelle Pinnacle-Closing-Odds für 1X2 oder exakte Ergebnisse
systematisch schlägt — und welche Backlog-Punkte lohnen sich danach noch?

**Methode:** Multi-Agent-Recherche über 5 Suchwinkel (Markt-Effizienz,
Exact-Score-SOTA, Contest-/Pool-Strategie, Kicktipp-spezifisch, Datenquellen).
20 Quellen gefetcht, 98 Claims extrahiert, Top 25 adversarial verifiziert
(3-Voter-Panel): 21 bestätigt, 4 widerlegt.

**Kernbefunde:**

1. **Kein verifizierter Closing-Line-Edge im Fußball.** Alle publizierten
   „Edges" (Wilkens 2026, Boshnakov et al. 2017, Egidi et al. 2018) wurden
   nur gegen weiche/durchschnittliche oder Nicht-Closing-Quoten gezeigt.
   Hubáček & Šír ([IJF 39(2), 2023](https://arxiv.org/abs/2010.12508))
   verifizieren, dass Pinnacle-Closing-Odds auf jedem Wahrscheinlichkeits-
   niveau unverzerrt sind; ihr eigener Profit entsteht durch *Dekorrelation*
   vom Markt (Arbitrage gegen die Buchmacher-Marge) — ein Mechanismus ohne
   Kicktipp-Analogon. Bestätigt unabhängig unser ~20–30-Pkt-Ceiling.

2. **Wilkens 2026 (isotone Recal): herabgestuft.** Die isotone Recalibration
   ist im Paper für fast den gesamten Profit verantwortlich (~1% → ~10% ROI),
   aber der Gewinn stammt aus der Korrektur eines *unkalibrierten rohen
   xG-Skellam-Modells* gegen Durchschnittsquoten von ~15 Soft-Buchmachern —
   „Pinnacle" kommt im Paper nicht vor. Unsere quoten-verankerte Matrix ist
   bereits kalibriert (ECE < 0.04, siehe Calibration-Test 2026-05-16).
   Erwartung: Null-Resultat. Der Headline-ROI-Claim des Papers hat die
   adversariale Verifikation nicht überlebt (1-2); das Paper selbst nennt
   seine Returns „an upper bound … rather than readily realisable profits".

3. **Egidi/Pauli/Torelli 2018 (Bayes-λ): gestrichen ohne eigenen Test.**
   Die Posteriors der Mischgewichte sind [im Paper selbst](https://arxiv.org/pdf/1802.08848)
   unidentifiziert (50%-Bars ≈ Prior, über 2.754 BL-Spiele 2007–2016) —
   spiegelt exakt unser λ-Sweep-Null (p=0.24). Das Modell verliert zudem in
   allen vier getesteten Ligen gegen quoten-implizite Wahrscheinlichkeiten
   (Bundesliga: 0.4010 vs. 0.4100 Shin). Die Profit-Claims des Papers:
   nur gegen 7 Soft-Buchmacher, ±1-s.e.-Bars bis Null.

4. **Exact-Score-SOTA: kein Upgrade über die aktuelle Matrix.** Zehn-Saison-
   Benchmark ([penaltyblog 2025](https://pena.lt/y/2025/03/10/which-model-should-you-use-to-predict-football-matches/),
   Eredivisie, RPS): zeitgewichtetes Dixon-Coles schlägt bivariates Poisson
   (schlechtestes Modell im Feld), Zero-Inflated, NegBin und Weibull-Copula.
   Boshnakov et al. (IJF 2017) profitieren nur auf 1X2/Totals gegen
   Soft-Durchschnittsquoten; der Claim „Weibull-Counts fitten Scores besser
   als Poisson" wurde widerlegt (1-2).

5. **Spielstrategie ist der einzige theoretisch fundierte, ungetestete Hebel.**
   Contest-Theorie: P(Runde gewinnen) ≠ erwartete eigene Punkte
   ([Clair & Letscher 2007](https://www.stat.berkeley.edu/~aldous/157/Papers/clair.pdf),
   Operations Research). In kleinen Pools konvergiert optimales Spiel aber
   gegen EV-max — Genauigkeit der Wahrscheinlichkeiten schlägt dort
   Opponent-Modeling. Standings-abhängige Varianz-Modulation
   ([Tsetlin/Gaba/Winkler 2004](https://link.springer.com/article/10.1023/B:RISK.0000038941.44379.82),
   J. Risk & Uncertainty): moderater Rückstand spät in der Saison →
   varianzreichere/unpopuläre exakte Ergebnisse; aussichtsloser Rückstand →
   *nicht* zocken (Konzessions-Resultat, empirisch Genakos & Pagliero);
   Führung → Feld spiegeln / EV-max. Die Volksregel „wer hinten liegt, muss
   zocken" ist damit in beide Richtungen falsch. Keine publizierte Analyse
   des Kicktipp-1/2/3-Schemas gefunden — die 2:1-vs-1:0-Frage ist nur per
   eigener Simulation beantwortbar.

6. **Datenquellen:** football-data.co.uk liefert kostenlos Pinnacle
   Pre-Closing- (PSH/PSD/PSA) *und* Closing-Spalten (PSCH/PSCD/PSCA), 1X2
   ab mind. 2018/19 — Cross-Check/Backfill für The Odds API. Achtung
   (widerlegt 0-3): Pre-Closing ≠ Opening; echte Opening→Closing-Drift
   braucht The Odds API-Snapshots (5-Min-Raster seit 09/2022, Credit-teuer)
   oder TheStatsAPI (~$50/Monat). Keine verifizierte Evidenz, dass Drift
   über die Closing-Line hinaus Information trägt. Keine freie Quelle für
   historische Pinnacle-Correct-Score-Quoten gefunden.

**Aktion:** Backlog neu priorisiert (siehe unten). Bayes-λ gestrichen,
isotone Recal herabgestuft, Varianz-Strategie-Simulation neu auf #1.

---

## 2026-08-16 — Diebold-Mariano-Test: Modell vs. Pinnacle-Closing auf RPS

**Frage:** Schlägt Pinnacle-Closing das DC-Modell auf einem *propren* Scoring
signifikant — und warum zeigt das Kicktipp-Punktemaß keinen Unterschied?

**Anlass:** Ein Modell-vs-Odds-Vergleich auf Kicktipp-Punkten wirkte „gleichauf".
Zwei Ursachen: (1) `fetch_odds_csv` benchmarkte gegen Pinnacle *Pre-Closing* (PSH)
statt Closing (PSCH) — behoben, Closing ist jetzt Default. (2) Kicktipp-Punkte sind
ein unpropres Scoring; die Exakt-Ergebnis-Form ziehen Modell und Odds aus derselben
`odds_to_score_matrix`, was den echten 1X2-Edge des Marktes verwischt.

**Methode:** RPS (proper, geordnetes 1X2) je Spiel für Modell und Closing, 1186
Spiele (2022–2025), season-basierter Split (`training_split`). Diebold-Mariano
(Lag 0 — Spiele unkorreliert) auf d = RPS_Modell − RPS_Closing; Paired Bootstrap
(n=10.000) als verteilungsfreier Cross-Check. Repro: `python backtest_dm_test.py`.

**Ergebnis:**

| Metrik | RPS Ø (niedriger = besser) |
|---|---|
| Modell (DC) | 0.2008 |
| Pinnacle Closing | 0.1972 |
| Kombiniert λ=0.7 | 0.1971 |

DM = +2.07, **p = 0.039**, Bootstrap-95%-CI [+0.0003, +0.0071] (schließt 0 aus).
Zum Kontrast Kicktipp-Punkte Ø/Spiel: Modell 0.810 vs. Closing 0.819 — statistisch
*nicht* unterscheidbar, Ranking kippt saisonweise.

**Nachtrag 2026-08-21 (leak-frei + dependence-aware, Review-Finding 4):** Nach dem
BL2-Leak-Fix ist das RPS unverändert (Modell 0.2008 / Closing 0.1972 — der Leak wirkte
praktisch nicht aufs 1X2-RPS). Die Lag-0-Annahme (unkorrelierte Spiele) wurde durch
Newey-West-HAC (Lag ≈ 1 Spieltag) und Moving-Block-Bootstrap ersetzt: **p=0.035 (HAC)
bzw. p=0.043 (Block-Bootstrap)** gegenüber iid p=0.039. Die Signifikanz **hält also
auch dependence-aware** (die HAC-Varianz fiel sogar minimal kleiner aus) — die Sorge,
ein abhängigkeits-robuster SE könne p über 0.05 schieben, bestätigt sich nicht.

**Befund:** Closing schlägt das Modell auf 1X2 **signifikant** — der Markt-Edge ist
real, aber so klein, dass ihn nur ein propres Maß sichtbar macht; auf Kicktipp-
Punkten verschwindet er. Erstmals das ~20–30-Pkt-Ceiling mit formalem
Signifikanztest untermauert. Kombiniert ≈ Closing: das Modell trägt über die
Closing-Line hinaus ~nichts zur 1X2-Prognose bei (Markteffizienz). Ausnahme:
Saison 25/26 schlug das Modell Closing auch auf RPS (0.1930 vs 0.1981) —
verrauschte, für alle Tipper überdurchschnittlich treffsichere Saison.

**Aktion:** `fetch_odds_csv` nutzt jetzt Closing (PSCH) als Default (Fallback →
PSH → generisch; `closing=False` erreichbar). Achtung: alle Backtest-Zahlen vor
2026-08 wurden gegen die schwächere Pre-Closing-Line gemessen — Reruns weichen ab.

---

## 2026-08-16 — λ-Sweep gegen Closing: bleibt ODDS_WEIGHT=0.7 optimal?

**Frage:** Verschiebt die Umstellung auf die (bessere) Closing-Line das optimale
Mischgewicht Modell/Odds weg von 0.7? Trigger: der DM-Test zeigt Closing > Modell.

**Methode:** Walk-forward λ-Sweep (0.0–1.0, Schritt 0.1) auf Kicktipp-Punkten,
918 Spiele (2022–2024), Closing-Odds. Paired Bootstrap (bestes λ vs. 0.7).
Repro: `python backtest_lambda_sweep.py`.

**Ergebnis:**

| λ (Odds-Gewicht) | Ø/Spiel | Δ vs 0.7 |
|---|---|---|
| 0.5 | 0.813 | −3 |
| 0.7 (Produktion) | 0.816 | ±0 |
| 0.8 | 0.824 | +7 |
| 0.9 / 1.0 | 0.827 | +10 |

Optimum bei λ=0.9–1.0, aber best−0.7 = +0.011 Pkt/Spiel, 95%-CI [−0.012, +0.035],
**p = 0.398** — nicht signifikant. Kurve von 0.2–1.0 praktisch flach; bestes λ pro
Saison instabil (0.7 / 1.0 / 0.9).

**Befund:** Das Optimum driftet richtungskonsistent zum DM-Befund nach oben (mehr
Odds-Gewicht, weil Closing das Modell schlägt), aber der Vorteil bleibt im
Rauschen — ODDS_WEIGHT=0.7 ist *nicht* signifikant geschlagen. Der Trainingsfilter-
Fix ist irrelevant für λ (Sweep nutzte immer den korrekten season-Split).

**Vorbehalt:** Sweep auf CSV-Closing; die Produktion tippt mit den-odds-api-Live-
Quoten. Das produktions-optimale λ ist ohne gespeicherte Live-Feed-Historie nicht
direkt messbar — dieser Sweep ist der beste verfügbare Proxy. 25/26 nicht im Test.

**Aktion:** Keine. ODDS_WEIGHT bleibt 0.7 (verteidigbar; falls überhaupt, minimal
auf 0.8 — theoretisch gestützt, aber im Rauschen).

---

## 2026-08-16 — Headroom der Exakt-Ergebnis-Schicht: lohnen Correct-Score-Quoten?

**Frage:** ~30 % der Kicktipp-Punkte stammen aus exakten Ergebnissen (Punkte-
Zerlegung: 96/1186 Exakt-Treffer = 8,1 %, 288 Pkt = 29,8 %), und diese Dimension
bekommt aktuell null Markt-Input — die Score-Verteilung erzeugt
`odds_to_score_matrix` rein aus 1X2 + O/U-2.5. Wie viel könnten Correct-Score-
Quoten maximal bringen?

**Methode:** Der Score-Layer kann die Tendenz nicht verbessern (Markt-Job), nur
das Ergebnis *innerhalb* der committeten Tendenz. Leiter von Decken, 1186 Spiele
(2022–2025). Repro: `python backtest_score_headroom.py`.

**Ergebnis (Ø Pkt/Spiel):**

| Strategie | Ø/Spiel |
|---|---|
| Naiv „immer 2:1" | 0.669 |
| Aktuell (Modell+Closing) | 0.816 |
| Beste konst. Scoreline / Tendenz (Hindsight) | 0.831 |
| Perfektes Ergebnis \| Tendenz fix | 1.606 |
| Absolut (tatsächliches Ergebnis) | 3.000 |

- Realistischer Headroom (populations-CS-Info): **+0.015 Pkt/Spiel (+18 Pkt über
  4 Saisons)** — und das ist Hindsight/in-sample, also optimistisch.
- Absolute Score-Layer-Decke: +0.79 Pkt/Spiel (+937) — aber unerreichbar (setzt
  das *tatsächliche* Ergebnis voraus, keine Verteilung).

**Befund:** Correct-Score-Quoten lohnen **nicht**. Die große Score-Layer-Decke
(+0.79) ist fast vollständig *irreduzibles Spiel-Rauschen* — exakte Ergebnisse sind
jenseits der Verteilung nicht prognostizierbar; keine Quote holt das. Der Teil, den
CS-Quoten realistisch liefern (populations-typische Score-Form), steckt bereits im
gut kalibrierten DC-Layer (ECE≈0,034, Calibration-Test 2026-05-16) → nur +0.015 in
Hindsight, und die OOS-Variante davon (Recalibration 2026-05-10) ging mit −27 bis
−35 Pkt sogar negativ. Dazu die Datenlage: keine freie historische Pinnacle-CS-
Quelle, Retail-CS-Märkte hochmargig/verrauscht.

**Aktion:** CS-Quoten-Thread geschlossen — kein verwertbarer Hebel.

---

## 2026-08-16 — Varianz-Strategie kalibriert: hilft „Zocken bei Rückstand"?

**Frage (Backlog #1):** Ab welchem Rückstand / wie vielen Restspieltagen schlägt
varianzreiches Tippen (γ: Tipp = argmax(EV + γ·Std)) die EV-Maximierung auf
P(Runde gewinnen)? Betrifft Strategie bei *fixen* Wahrscheinlichkeiten.

**Methode:** Monte-Carlo, DGP = DC-Score-Matrizen (`training_split`). Heterogenes
20er-Feld (jeder Gegner mit eigener Softmax-Temperatur T_i), kalibriert an
öffentlichen Punkte-Ankern: EV-max ≈ 0.83, naiv „2:1" = 0.669 Pkt/Spiel. Drei
Feldstärken. Repro: `python backtest_variance_strategy.py`
(`--calibrate` für die T→Punkte-Abbildung; `--field sharp|mixed|casual`).

**Ergebnis (ΔP(win) von γ* gegenüber EV-max):**

| Feld (Ø Pkt/Spiel) | Wo hilft Varianz? | max ΔP |
|---|---|---|
| **sharp** (0.79) | Rückstand + wenige Restspiele → γ=2 | **+0.026** |
| mixed (0.68) | nur extremer Rückstand, marginal | +0.005 |
| **casual** (0.63) | nirgends — EV-max dominiert überall | +0.003 |

**Befund:** Die Varianz-Strategie hat einen **echten, theoriekonformen Edge — aber
nur gegen ein scharfes Feld** (Gegner spielen selbst nahe EV-max; dann muss man
zocken, um zu überholen). Gegen ein **casual Feld — genau die reale Runde (85 %
unter EV-Baseline) — hilft Varianz nie**; EV-max dominiert in jeder Rückstand-/
Restspiel-Zelle. Als scharfer EV-max-Tipper gewinnt man ein casual 20er-Feld schon
per reiner Genauigkeit >50 % (P=0.57 bei R=10, d=0); jede Varianz senkt das nur.
Die Volksregel „wer hinten liegt, muss zocken" ist für casual Runden falsch. Nicht
die Pool-*Größe* entscheidet (Literatur-Review Punkt 5), sondern die Pool-*Stärke*.

**Vorbehalt:** Getestet gegen ein Softmax-über-Modell-EV-Feld. Eine
*Anti-Popularitäts*-Strategie (bewusst weg von populären Ergebnissen wie 2:1/1:0,
wenn das Feld darauf klumpt) ist ein anderer Hebel, den dieses Feldmodell nicht
erfasst — er bräuchte echte Tippverteilungen (ligenintern, bleiben lokal).

**Aktion:** Backlog #1 praktisch geschlossen — in casual Runden bei EV-max bleiben.
Offen nur die Anti-Popularitäts-Variante (braucht private Tippdaten).

---

## 2026-08-16 — Remis-Bias: untertippt das Modell Unentschieden? (Null-Resultat)

**Frage:** Im Punkteschema zahlt eine korrekte Remis-*Tendenz* 2 Punkte, eine
Sieg-Tendenz nur 1 — Remis sind doppelt wertvoll. DC-Poisson unterschätzt in der
Literatur Unentschieden. Tippt EV-max dadurch zu selten Remis und verschenkt Punkte?

**Methode:** Kalibrierung (Ø P(Remis) vs. reale Quote) + Remis-Boost: Diagonale der
Produktions-Score-Matrix mit δ skalieren, renormieren, EV-max neu, Kicktipp-Punkte
vergleichen. 1186 Spiele (2022–2025). Repro: `python backtest_draw_bias.py`.

**Ergebnis:**
- Ø P(Remis) = **0.236** vs. tatsächlich **0.250** — nur minimal unterschätzt (ρ-Korrektur wirkt).
- Modell tippt Remis in nur **4,9 %** der Spiele.

| δ (Remis-Boost) | Ø Pkt/Spiel | Δ vs 1.0 |
|---|---|---|
| **1.0** | **0.816** | ±0 |
| 1.15 | 0.803 | −0.013 |
| 1.3 | 0.815 | −0.001 |
| 1.5 | 0.794 | −0.022 |
| 2.0 | 0.728 | −0.088 |

**Befund:** δ=1.0 optimal — jeder Remis-Boost verschlechtert. EV-max wägt den
2-Punkte-Remis-Bonus bereits korrekt gegen die höhere Sieg-Wahrscheinlichkeit ab;
mehr Remis zu erzwingen kostet auf Nicht-Remis-Spielen mehr, als es auf Remis
einbringt. Die scheinbare „Remis-Schwäche" (wenig Punkte an Remis-Spielen) ist der
korrekte EV-max-Trade-off, kein ausnutzbarer Fehler. Kein Modell-Hebel.

---

## 2026-08-20 — Deep-Research-Nachfassen: Schlägt irgendetwas Pinnacle-Closing? (extern bestätigt)

**Frage:** Erweiterung des Literatur-Reviews vom 2026-07-14 — gibt es *neue* (2024–2026)
publizierte oder reproduzierbare Evidenz, dass ein Modell oder eine Datenquelle
Pinnacle-**Closing** für 1X2 oder exakte Ergebnisse schlägt? Fokus: proper scoring
(RPS/LogLoss/Brier) und echter CLV, moderne ML (GNN/Transformer/GBT), strikt gegen die
*scharfe* Closing-Line (Soft-/Durchschnitts-/In-Play-/Opening-Benchmarks zählen nicht).

**Methode:** Multi-Agent-Deep-Research über 5 Suchwinkel, 16 Quellen gefetcht, 66 Claims
extrahiert, Top 25 adversarial verifiziert (3-Voter-Panel, 2/3-Refute killt): **22 bestätigt,
3 widerlegt, 0 unverifiziert**.

**Kernbefunde:**

1. **Erstmals direkte Evidenz gegen Pinnacle-Closing — und sie ist negativ.**
   Pitcan 2026 (Serie A, [arxiv 2608.11505](https://arxiv.org/html/2608.11505)) ist der erste
   gefundene Aufsatz, der explizit die Closing-Line als Benchmark nutzt, mit fast unserem
   Setup: zeitgewichtetes **Dixon-Coles, log-pooled mit Pinnacle-Closing**. Das optimale
   Mischgewicht auf dem DC-Modell ist **exakt 0.000** (Rand-Lösung, auf Validierung *und* Test
   bestätigt, LogLoss monoton steigend im Gewicht). Closing schlägt DC auf allen proper scores
   (n=2.660): **RPS 0.1905 vs. 0.1972**, LogLoss 0.962 vs. 0.986, Brier 0.572 vs. 0.586
   (paired RPS-Diff +0.0067, 95%-CI [0.0046, 0.0088]). O-Ton: *„the closing price has already
   absorbed both."* → **Externe Replikation unseres DM-Tests** (2026-08-16: unser Modell 0.2008
   vs. Closing 0.1972): gleiche Richtung, gleiche Größenordnung, andere Liga.

2. **Reproduzierbarer CLV-Test, ebenfalls negativ.** Boui-Repo
   ([github.com/zakariae-boui/football-prediction-ml](https://github.com/zakariae-boui/football-prediction-ml)):
   Wetten zu Bet365, bewertet gegen Pinnacle-Closing (de-vigged), 6.080 PL/LaLiga-Spiele mit
   Understat-xG → **negativer CLV für alle Modelle** (RF/XGBoost/SVM). Kein zirkulärer
   CLV-Fehler (Closing nur zur Bewertung, nie als Feature).

3. **Moderne ML schlägt den Markt nicht — und testet meist gar nicht dagegen.**
   Graph-Transformer **HIGFormer** ([2507.10626](https://arxiv.org/abs/2507.10626), WyScout-Events)
   benchmarkt nur gegen andere ML-Modelle, nie gegen Quoten, nur 52,2 % Accuracy, kein RPS/CLV.
   Der Deep-Learning-Sieger der 2023 Soccer Prediction Challenge
   ([Springer](https://link.springer.com/article/10.1007/s10994-024-06608-w)) **verliert** gegen
   Buchmacher-Konsens auf RPS (0.2195 vs. 0.2063). Der ML-Sports-Betting-Review
   ([2410.21484](https://arxiv.org/html/2410.21484v1)) erwähnt „closing"/„Pinnacle"/CLV **kein
   einziges Mal**.

4. **Alle 2024–26 „Profit"/„Score-Win"-Headlines fallen am scharfen Closing durch.**
   Wilkens 2026 (Bundesliga, [JSA](https://journals.sagepub.com/doi/10.1177/22150218261416681)):
   Benchmark = Ø **~15 Soft-Books** + Line-Shopping, Returns explizit „upper bound … rather than
   readily realisable"; auf proper scores gewinnt der *Markt* den Brier. AFT
   ([2605.16066](https://arxiv.org/abs/2605.16066)) = **In-Play**-Betfair, nicht Pre-Match/Closing,
   unterbietet den Markt sogar auf Accuracy. Egidi 2018 = **7 Soft-Books** (verliert dennoch gegen
   odds-implizit in jeder Liga — Overfit-Widerspruch). Hegarty & Whelan 2024
   ([IJF](https://www.sciencedirect.com/science/article/pii/S0169207024000670)) = Soft-Book-AH,
   kein Head-to-Head gegen Pinnacles eigene 1X2-Closing.

5. **De-Vig cappt per Konstruktion, Exact-Score bleibt datenarm.** MDPI 2025
   ([Mathematics 13(24):3976](https://www.mdpi.com/2227-7390/13/24/3976)) sagt selbst: bestenfalls
   *Parität* mit dem Buch, nie Gewinn (Marge). Exact-Score: weiter **keine** CLV-Evidenz; der
   LLM-Rerank-Harness ([2608.05030](https://arxiv.org/html/2608.05030)) benchmarkt nie gegen
   Quoten; weiterhin keine freie Pinnacle-Correct-Score-Quelle → CS-Ceiling bleibt nur *inferiert*
   (konsistent mit Headroom-Analyse 2026-08-16).

**Widerlegt (adversarial, für Transparenz):** „scharfer AH-Markt beweisbar bias-frei" (1-2),
Stübinger et al. 2019 RF-„Edge" (~1,58 %/Spiel; 1-2, kein Signifikanz-/Sharp-Line-Test),
„margin-removed Pinnacle-Closing perfekt kalibriert in 1%-Bins" (0-3, Blog-Qualität).

**Aktion / Implikation:**
- Das ~20–30-Pkt-Ceiling ist jetzt **extern und in unserem Setup** bestätigt (DC + Closing →
  Gewicht 0.000). „Match the closing line" ist das empirische Ceiling, kein bloßer Prior.
- DC-Layer bleibt **Fallback** (fehlende/stale Closing, Exact-Score-Verteilung), kein
  Informations-Add-on. Isotone Recal (Backlog #3) und GAS/pi-Rating (#4) unverändert erwartetes
  Null → Priorität bleibt gesenkt.
- Einziger belegter Hebel bleibt **Kicktipp-Score-Optimierung auf** der Closing-Line (Backlog #1,
  Anti-Popularitäts-Variante) — Entscheidungstheorie, nicht bessere Wahrscheinlichkeiten.

**Vollständiger Report** (alle 6 Findings, Caveats, 4 offene Fragen, 16 Quellen):
[docs/research/2026-08-20-closing-line.md](docs/research/2026-08-20-closing-line.md).

---

## 2026-08-20 — Exact-Score-Proxy aus Asian Handicap + Totals (verworfen)

**Hypothese:** Die 1X2-Rekonstruktion (`odds_to_score_matrix`, 2-Param-Poisson,
2 Freiheitsgrade) lässt die Score-Verteilung unterbestimmt. Pinnacle-Closing
**Asian Handicap** (fixiert die Tordifferenz) und **Over/Under** (fixiert die
Gesamttore) könnten die exakte Ergebnisverteilung schärfer pinnen → mehr 3-/2-Punkte-
Treffer, Hebung des ~231-Ceilings Richtung theoretischer 237–252 (offene Frage #3
aus dem Deep-Research-Nachfassen desselben Tages).

**Datenlage:** football-data.co.uk liefert alle Pinnacle-*Closing*-Spalten
offline: 1X2 (PSCH/PSCD/PSCA), O/U 2.5 (PC>2.5/PC<2.5), Asian Handicap
(AHCh + PCAHH/PCAHA). Kein Team-Mapping nötig — Ergebnis steht in derselben Zeile.

**Stufe 1 — Diagnose** (`backtest_score_markets.py`, 1.985 Spiele 2019–2025):
Trägt AH/OU Info über die 1X2-Poisson-Matrix hinaus? Test = Brier(Markt) vs.
Brier(1X2-Modell) auf realen Ausgängen.
- **Asian Handicap (Tordifferenz):** 1X2-Matrix trifft die AH-Preise auf **MAE
  1,4pp** und ist auf realen Ausgängen **nicht unterscheidbar** vom Markt
  (Brier 0.2495 vs. 0.2478, Δ n.s.). AH-Linien sind bewusst um ~0.5 balanciert →
  fast Münzwurf, kaum marginale Info jenseits 1X2.
- **Over/Under 2.5 (Gesamttore):** Markt schlägt signifikant (Brier 0.2282 vs.
  0.2437); das 1X2-Modell **untertippt Overs um 9,6pp**. Info existiert — aber auf
  der für Kicktipp weniger relevanten Achse.

**Stufe 2 — Definitiv-Test** (`backtest_score_proxy.py`, 915 Spiele mit vollem
Closing): Dixon-Coles-Fit mit ρ, gemeinsam auf 1X2+O/U+AH, EV-optimaler Tipp,
gepaart gegen 1X2-only.

| Rekonstruktion | Pkt | Ø/Spiel |
|---|---|---|
| A  1X2-only | 716 | 0.7825 |
| B  1X2 + O/U | 732 | 0.8000 |
| C  1X2 + O/U + AH | 727 | 0.7945 |

- Δ B−A (O/U) = +16 Pkt (+0.0175), **n.s.** [−0.014, +0.049]
- Δ C−B (AH)  = −5 Pkt (−0.0055), **n.s.**
- **Δ C−A (Proxy gesamt) = +11 Pkt (+0.0120), n.s.** [−0.036, +0.061]

Trotz hoher Tipp-Churn (A→B 49,7 %, B→C 38,7 % geänderte Tipps) ist der Netto-Effekt
statistisch null — die Änderungen fallen in Situationen mit verschwindendem EV-Gap
(deckt sich mit EV-Gap-Test 2026-05-16: 88,7 % der Disagreements bei Gap <0.01).

**Befund / Aktion:** Der Exact-Score-Proxy bringt auf Kicktipp-Punkten **keinen
signifikanten Gewinn**; die AH-Achse (die relevante) trägt nichts über 1X2 hinaus.
Damit ist die ~231-Ceiling-Frage *mechanistisch* beantwortet: nicht CS-Info fehlt,
sondern 1X2 pinnt die Tordifferenz bereits so scharf wie der AH-Markt. **Verworfen.**
Der Punkt-Schätzer (+11) liegt am unteren Rand des theoretischen +5–20-Headrooms, ist
aber nicht von 0 unterscheidbar und rechtfertigt keine fragile Fit-Maschinerie.
Nebenbefund: O/U ist auf *Closing* neutral (n.s.), nicht −11 wie das ältere
Pre-Closing-Ergebnis in der README-Tabelle — Reruns weichen ab (siehe 2026-08-16).

**Reproduktion:** `python backtest_score_markets.py` (Diagnose) ·
`python backtest_score_proxy.py` (Definitiv-Test).

---

## 2026-08-21 — Temporaler BL2-Leak im Trainings-Split gefixt (externer Review)

**Befund (unabhängiger Review):** `training_split` schloss nur die laufende
*BL1*-Saison ab dem Spieltag aus und behielt die **gesamte laufende BL2-Saison** —
auch Spiele, die relativ zum vorhergesagten BL1-Spieltag chronologisch in der
*Zukunft* liegen. Über das `time_weight`-Clamping (Zukunft → Alter 0 → Gewicht 1.0)
bekamen diese künftigen BL2-Spiele sogar **maximales Gewicht**. Echter temporaler
Leak (2. Ordnung: BL2-Teams sind nicht im BL1-Spieltag, kontaminieren aber die
geteilten Parameter γ/ρ und künftige Aufsteiger-Ratings).

**Fix:** `training_split` bekommt einen optionalen `ref_date`-Datums-Cutoff
(`date >= ref_date` → raus), durchgereicht von jedem Produktions-/CLI-Aufrufer
(dasselbe `ref_date` wie `fit_dixon_coles`). Schließt zusätzlich verschobene/
vorverlegte Spiele korrekt. Der Cutoff vergleicht echte datetime, NICHT
`date.year` (der alte Bug bis 2026-08). Regressionstest ergänzt.

**Effekt (SP1–30 2024/25, 1X2-only wie `cmd_backtest`):**

| Modus | vorher (kontaminiert) | nachher (leak-frei) |
|---|---|---|
| Nur Modell | 211 / 0.781 | **206 / 0.763** |
| Modell + Quoten | 225 / 0.833 | **228 / 0.844** |

Größenordnung wie erwartet klein (Einzelsaison-Rauschen; Modell-only −5, Blend +3
— beide könnten saisonweise das Vorzeichen wechseln), aber die Vorher-Zahlen waren
*kontaminiert*. Richtung (Markt ≥ Modell) unverändert.

**Noch offen (nächste Schritte des korrektiven Passes):** (a) Forecast-Stack
vereinheitlichen — `cmd_backtest` strippt O/U, Produktion nutzt es; die
Experiment-Skripte (disagreement/λ-sweep/dm-test/…) tragen denselben Leak und
werden mit ihrer Neurechnung gefixt; (b) Reverse-Fixture-Odds-Fallback entfernen;
(c) danach die Validierungs-Zahlen (Disagreement 142/15.5 %, DM p=0.039) mit
dependence-aware SE neu rechnen. Bis dahin bleiben die Zahlen im README-Abschnitt
„Statistische Validierung" die alten.

---

## 2026-08-21 — Forecast-Stack vereinheitlicht + Validierungs-Zahlen leak-frei neu gemessen

**Kontext (Korrektur einer eigenen Fehlaussage):** Der Review vermutete, `cmd_backtest`
strippe O/U und weiche damit von der Produktion ab. Verifiziert: Produktion ist selbst
**1X2-only** — `fetch_live_odds` berechnet O/U, packt es aber nicht in den Odds-Dict
(Z. 517: „O/U wird abgerufen aber nicht genutzt"). Prod und `cmd_backtest` sind also
konsistent (bewusst 1X2-only). Die einzige Drift lag in den **Experiment-Skripten**
(disagreement/ev-gap/calibration/λ-sweep), die über `fetch_odds_csv` O/U mitzogen.

**Fix:** Diese Skripte auf 1X2-only gezogen (kanonischer Stack) **und** den BL2-Leak
gefixt (ref_date-Cutoff wie in kicktipp.py), dann neu gerechnet.

**Neue Zahlen (leak-frei + 1X2-konsistent, 918 BL1-Spiele 2022–2024) — Schlüsse halten:**
- **Disagreement:** 142/15,5 % → **182/19,8 %** (O/U-Strip lässt Modell & Quoten öfter
  divergieren). λ=0.7: p=0.72 → **0.775**; λ=0.3: p=0.59 → **0.875**. Beide klar n.s. —
  kein DC-Edge, robuster als vorher (der λ=0.3-„Vorteil" ist jetzt ~0).
- **EV-Gap:** „88,7 % der Disagreements bei Gap <0.01" → **77,5 %**; „bei Gap ≥ 0.04
  100 % Übereinstimmung" hält (0 Disagreements). Pro Bin weiter keine konsistente Richtung.
- **Calibration:** Auf **1X2** weiter gut kalibriert (ECE < 0.03, Mix am besten). Auf
  **Over/BTTS** jetzt schlecht (ECE 0.08–0.11), weil die 1X2-only-Rekonstruktion Overs
  ~11pp untertippt — konsistent mit der Score-Markets-Diagnose; für die Tipp-Wahl
  (argmax über 0:0–2:2) folgenlos.

**Erledigt 2026-08-21:** `backtest_dm_test.py` (dependence-aware, siehe Nachtrag oben)
sowie λ-sweep/draw-bias/score-headroom/variance/recalibration — alle leak-frei +
1X2-konsistent neu gerechnet, **alle Schlüsse halten**: λ-Sweep bestes λ=0.5 aber n.s.
gegen 0.7 (Saison-Optima 0.4/0.9/0.6 → kein robustes Optimum, ODDS_WEIGHT=0.7 bleibt);
draw-bias δ=1.0 optimal (kein Remis-Edge); score-headroom Δ(3−2)≈+0.014 Pkt/Spiel (~0);
variance-tilt γ*≈0 im kalibrierten Casual-Feld; recalibration n.s. negativ (bleibt
verworfen). Reverse-Fixture-Fallback in `_find_odds` entfernt (Finding 2): Headline-
Backtest 228 unverändert → der Zweig war dormant, feuerte nie, aber die stille
Falsch-Substitution ist als Footgun beseitigt. **Härtung erledigt (Findings 5–8):**
Timezone auf `zoneinfo.ZoneInfo("Europe/Berlin")` (Winter-Anpfiffzeiten waren ~5 Monate/
Saison 1 h falsch); `fit_dixon_coles`-Convergence-/Finitheits-Guard + ρ-Clamp (feuerte
im Headline-Backtest nie → rein defensiv, 228 unverändert); `fetch_season`-TTL für die
laufende Saison (abgeschlossene bleibt permanent); `submit_tips` verifiziert nach der
Abgabe per Formular-Refetch, dass die Tipps wirklich gespeichert sind. **Damit ist der
komplette korrektive Pass (Review-Findings 1–8) abgeschlossen — jeder inhaltliche
Schluss hält.**

---

## Backlog (aktualisiert 2026-08-21)

1. ✅ **Standings-abhängige Varianz-Strategie** (weitgehend erledigt 2026-08-16) —
   `backtest_variance_strategy.py` mit kalibriertem, heterogenem Feld: Varianz-Tilt
   schlägt EV-max nur gegen ein *scharfes* Feld; in casual Runden (unsere) hilft er
   nie → bei EV-max bleiben. Offen nur die **Anti-Popularitäts-Variante** (bewusst
   weg von populären Ergebnissen) — braucht echte Tippverteilungen (ligenintern,
   bleiben lokal). Siehe Eintrag oben.
2. ✅ **football-data.co.uk Pinnacle-Spalten** (erledigt 2026-08-16) —
   `fetch_odds_csv` nutzt jetzt die Closing-Line (PSCH) als Default; der
   DM-Test oben zeigt, dass Closing das Modell signifikant schlägt.
   Nachzug 2026-08-21: Auch die Over/Under-Rekonstruktion nutzt jetzt Closing
   (`PC>2.5`, Fallback → `P>2.5`) statt Pre-Closing. **Korrektur (2026-08-21):**
   Der damalige „225 → 225, null Effekt"-Nachweis war *irreführend* — die Zahl
   bewegte sich nicht, weil Closing-O/U neutral wäre, sondern weil `cmd_backtest`
   O/U ohnehin **strippt** (Z. ~1361, nur 1X2). Dass O/U auf Kicktipp-Punkten
   ~neutral ist, gilt separat (Proxy-Test 2026-08-20, B−A n.s.), aber der
   Headline-Backtest nutzt es gar nicht — Produktion dagegen schon. Diese
   Stack-Inkonsistenz wird im nächsten Schritt behoben (Eintrag 2026-08-21 unten).
3. **Isotone 1X2-Recalibration** (Wilkens 2026) — nur noch als billiger
   Bestätigungstest (isotoner Fit auf Rolling-Window), erwartetes Ergebnis:
   Null (siehe Literatur-Review Punkt 2). Schließt den Punkt so oder so.
4. **Score-driven Team-Stärken (GAS)** (Koopman & Lit 2015/2019) und
   **pi-Rating** (Constantinou 2013) — bleiben als Modell-Layer-Ideen,
   aber Priorität gesenkt: das ~20–30-Pkt-Ceiling (λ-Sweep, EV-Gap-Test,
   Literatur-Review Punkt 1, extern repliziert 2026-08-20: Pitcan Serie-A,
   DC-Pooling-Gewicht 0.000 gegen Closing) deckelt den Nutzen jeder DC-Verbesserung.

**Gestrichen:** Bayesianische λ-Schätzung (Egidi/Pauli/Torelli 2018) —
Begründung im Literatur-Review vom 2026-07-14, Punkt 3.
**Gestrichen:** Correct-Score-Quoten — Headroom-Analyse 2026-08-16 zeigt
~0 realistischen Hebel (irreduzibles Score-Rauschen + gut kalibrierter DC-Layer);
zusätzlich direkt widerlegt 2026-08-20 (Exact-Score-Proxy aus AH+Totals, C−A n.s.,
`backtest_score_proxy.py`).
