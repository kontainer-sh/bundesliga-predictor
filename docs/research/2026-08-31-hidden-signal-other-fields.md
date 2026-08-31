# Versteckte Signale aus fachfremden Feldern — schlägt irgendetwas die Closing-Line?

**Datum:** 2026-08-31
**Frage:** Gibt es Ansätze *jenseits* von Standard-Sportstatistik/ML — aus Netzwerkwissenschaft,
Signalverarbeitung/Regelungstechnik, Marktmikrostruktur/Ökonometrie, statistischer Physik,
Informationstheorie und Spieltheorie —, die aus der Zeitachse- und Korrelationsstruktur von
Ergebnis- und Quotendaten un-eingepreistes prädiktives Signal ziehen?
**Methode:** Fan-out-Deep-Research (Web-Suche je Feld → Quellen fetchen → Claims extrahieren →
adversariale 3-Stimmen-Verifikation, 2/3-Refute killt → Synthese). 22 Quellen, 97 Claims,
25 verifiziert (21 bestätigt, 4 gekillt).

## Kontext / Benchmark

Das System ist ein zeit-gewichtetes Dixon-Coles-Modell (30 %) log-linear geblendet mit
Pinnacle-Quoten (70 %), reduziert auf den Tipp, der die erwarteten Kicktipp-Punkte (1/2/2/3)
maximiert. **Harte Messlatte:** Der eigene Diebold-Mariano-Test (Newey-West HAC p=0.035,
Block-Bootstrap p=0.043; extern repliziert durch Pitcan 2026, DC-Pooling-Gewicht mit Pinnacle-
Closing = exakt 0.000) zeigt: **Pinnacle Closing schlägt das Modell auf RPS**, und der realistische
Headroom *jedes* besseren Wahrscheinlichkeitsmodells ist ~20–30 Pkt/Saison. Wiederkehrender
Fehlermodus: „das Signal steckt schon in den Quoten". Jeder Ansatz wird daher gelabelt: (a) gegen
sharp/closing oder nur gegen soft/opening getestet? (b) plausibel schon eingepreist? (c) verbessert
er einen proper score (RPS/Brier) — oder die *Entscheidung* (Punkte/Scoreline) ohne bessere
Wahrscheinlichkeit?

## TL;DR — Verdikt je Feld

| Feld | Verdikt | Grund |
|---|---|---|
| **Spiel-/Entscheidungstheorie vs. Pool** | 🟢 einziger un-erschöpfter Hebel | verbessert die Entscheidung, braucht kein besseres Modell |
| **Marktmikrostruktur / Line-Movement** | 🟡 Ops-Timing ja, Modell nein | Closing schärfer → Snapshot später; Drift *modellieren* ~eingepreist |
| **Netzwerk / Graph** | 🔴 null | Edge kollabiert im vollständigen Round-Robin; nie gegen sharp getestet |
| **Signalverarbeitung / Kalman** | 🔴 null | State-Space ≈ globaler Exponential-Decay (Dixon selbst, 2002) |
| **Stat. Physik / Hawkes / Momentum** | 🔴 bestätigt null | Bundesliga-Studie 2025: Momentum sagt Sieger nicht vorher, Markt ignoriert korrekt |

**Kernbefund:** Kein einziger überlebender Ansatz wurde je gezeigt, Pinnacle Closing auf einem
proper score zu schlagen.

## 🟢 Angle 5 — Anti-Popularity (Entscheidungsseite)

Unter einem nicht-trivialen Punkteschema schlägt „erwartete Pool-Punkte maximieren + bewusst von
der Crowd differenzieren" das naive Favoriten-Tippen — eine wahrscheinlichkeits-*freie*
Entscheidungsverbesserung. Das 1/2/2/3-Exakt-Schema ist genau so eine Struktur.

- **Clair & Letscher (2007), *Operations Research* 55(6):1163–1177** — beweisen, dass in großen
  Pools Differenzierung vom Crowd-Pick „often by orders of magnitude" schlägt.
  <https://pubsonline.informs.org/doi/abs/10.1287/opre.1070.0448>
- **Kaplan & Garstka (2001), *Management Science* 47(3):369–382** — Modell nicht treffsicherer als
  Setzlisten (~58 %), aber EV-Optimierung schlägt sie *nur* unter nicht-trivialem Schema; unter
  „max. Anzahl richtig" ist der EV-optimale Tipp = Favorit (Hebel null by construction).
  <https://pubsonline.informs.org/doi/10.1287/mnsc.47.3.369.9769>

**Caveat:** Die großen Gewinne stammen aus Winner-take-all-Geldpools / Single-Elimination-Brackets.
Ein saisonlang kumulativer Casual-Pool ist strukturell anders — der Hebel-*Typ* überträgt sich, die
*Größe* nicht garantiert. Distinct von schon verworfenem Draw-Boost & Varianz-Tilt, weil andere
Zielgröße (Rang-vs-Feld statt Absolut-Punkte). Braucht die reale Tipp-Verteilung der Liga.

## 🟡 Angle 1 — Marktmikrostruktur / Line-Movement

Die eigentliche Erkenntnis ist kein Modellier-Hebel, sondern **Ops-Timing:** Backtest/Benchmark
nutzen Closing, die Produktion tippt mit Pre-Closing-Odds (The Odds API). Closing ist bewiesen
schärfer → der Spalt sind geschenkte Punkte, die nur am Abgabe-Zeitpunkt hängen. Fix: Odds so spät
wie möglich vor der Deadline ziehen (Kicktipp-Deadline = Anpfiff des 1. Spiels des Spieltags → nur
das 1. Spiel bekommt Near-Closing).

Die Drift zu *modellieren* lohnt dagegen nicht:
- NFL-Odds-Momentum existiert (Moskowitz, *Asset Pricing and Sports Betting*, *J. Finance*), aber
  eine *Management-Science*-Studie findet Überreaktion/negative Autokorrelation bei feinerer Frequenz.
- **betaminic (180.000+ Fußballspiele):** auf Bewegungsrichtung konditioniertes Setzen bringt keinen
  exploitierbaren Edge. → Drift plausibel schon in Pinnacle-Closing eingepreist.
- (Primärbeleg für die Momentum-These war nur eine Undergraduate-Thesis, Vote 2-1 → mittlere Konfidenz.)

## 🔴 Angle 2 — Netzwerk / Graph

Elegante, datenarme Methoden (nur Ergebnisse+Spielplan): Random-Walker/PageRank
(**Callaghan/Mucha/Porter**, arXiv physics/0310148), degree-neutralisierte Random Walks
(**Shin/Ahnert/Park, PLOS ONE 2014**, pone.0113685), dynamische Zentralität
(**Motegi & Masuda, Sci. Rep. 2012**). Aber:

1. Der Edge **kollabiert im vollständigen Double-Round-Robin** der Bundesliga (jeder gegen jeden →
   gleiche Grade; PLOS ONE misst dort nur ~61–62 % = Parität).
2. **Nie gegen sharp/closing getestet.** Wo ein Wett-Edge auftaucht (IJF 2019, „Efficiency of online
   football betting markets"), kommt er aus Line-Shopping (4,45 % ROI auf Best-of-41-Books vs. 2,78 %
   auf Mittel) — ein Soft-Book-/Favourite-Longshot-Artefakt, kein Sieg über Pinnacle. Lazova/Basnarkov
   (arXiv 1503.01331) validieren PageRank nur gegen FIFA-Ranking-Ähnlichkeit, nie gegen Ergebnisse/Odds.

*Mögliche dünne Ausnahme:* transitives Signal über die **BL1↔BL2-Grenze** (Auf-/Abstiegs-Kanten) für
frisch Aufgestiegene — verbindet sich mit dem Aufsteiger-Odds-Mapping-Thema. Niedrige Priorität.

## 🔴 Angle 3 — Signalverarbeitung / Kalman / State-Space

- **Crowder, *Dixon*, Ledford & Robinson (2002), *JRSS-D* 51(2):157–168** — ein dynamisches
  AR(1)/State-Space-Stärkemodell ist „at least competitive… though not demonstrably superior" ggü.
  statischem DC-mit-Likelihood-Tapering (~48–49 % beide). **D. h. der globale Exponential-Decay ist im
  Kern schon die „richtige" Antwort; ein Filter statt einer Halbwertszeit bringt ~0.**
  <https://academic.oup.com/jrsssd/article/51/2/157/7120674>
- **Koopman & Lit (2015), *JRSS-A* 178(1):167–186** — „significant positive return", aber nur gegen
  generische (soft) Odds, kein RPS/Brier gegen sharp.
- **Duffield/Power/Rimella (arXiv 2308.02414)** — State-Space schlägt nur Elo/Glicko/TrueSkill auf
  Log-Likelihood; „Pinnacle/Brier/RPS" kommen null Mal vor. (Behauptung eines fertigen Python-Toolkits
  „abile" wurde 0-3 gekillt.)

Deckt sich mit dem eigenen GAS/pi-rating-Downgrade; 0,001–0,03 Brier-Gewinn plausibel schon in Closing.

## 🔴 Angle 4 — Stat. Physik / Hawkes / Momentum

- **Ötting, Deutscher, Singleton & De Angelis (2025), *Economic Inquiry* 63(4)** — 1.224 Bundesliga-
  Spiele, sekündliche In-Play-Odds: „We find no evidence that the sequence of goals (who scored the
  equalizer) … affects the relative likelihood that one team will win." Wetter setzen ~40–60 % mehr auf
  das ausgleichende Team, Momentum-Following-Strategien liefern **−16 % bis −21 %** — der Markt ignoriert
  ein nicht-prädiktives Signal *korrekt*. Self-exciting/Hawkes-Features als Ergebnis-Prädiktor: tot.
  <https://onlinelibrary.wiley.com/doi/10.1111/ecin.70008>

## In der Verifikation gekillte Claims (Transparenz)

- „Markt besser kalibriert als simples xG-Skellam-Modell" → 0-3 gekillt.
- „xG-Modell macht positive ROI trotz schlechterer Kalibrierung" → 1-2 gekillt.
- „Fertiges Python-State-Space-Toolkit (abile) existiert" → 0-3 gekillt.
- „NFL-Closing nicht prädiktiver als Opening" (arXiv 1211.4000) → 0-3 gekillt (stützt eher die
  Closing-Schärfe-Prämisse).

## Meta-Muster & Fazit

Nahezu jeder „Edge" der Literatur wurde gegen soft/average/best-available/opening gemessen — **nicht**
gegen eine einzelne scharfe Pinnacle-Closing-Line; wo ein proper score berechnet wurde, gewann der
Markt. Die zwei lebenden Kandidaten umgehen die Wahrscheinlichkeitsfrage (Decision-side Pool-Play) oder
sind Ops-Timing (Line-Movement) — beide kein besseres Modell.

**Praktische Konsequenz:** Das Wahrscheinlichkeitsmodell ist ausgereizt. Verbleibende, ehrlich
un-erschöpfte Fäden: (1) Anti-Popularity gegen die Remis-Aversion des Feldes, rang- statt
punkte-optimiert — dünn, braucht Feld-Tipp-Verteilung; (2) Odds-Snapshot näher an die Deadline schieben
(gratis, null Modellrisiko). Alles andere ist gegen den ~20–30-Pkt-Deckel bestätigt null.

Siehe auch die quantitative Deckel-/Varianz-Analyse (Rest-Skill ~3 Pkt/Saison ≪ Zufalls-SD ~15) und
`docs/research/2026-08-20-closing-line.md`.
