# Does Anything Beat Pinnacle Closing Odds in Football? — 2024–2026 Evidence Update

> _Automatisch erzeugter Multi-Agent-Deep-Research-Report, 2026-08-20._
> _Methode: 5 Suchwinkel · 16 Quellen gefetcht · 66 Claims extrahiert · 25 adversarial verifiziert (3-Voter-Panel, 2/3-Refute killt) → 22 bestätigt / 3 widerlegt / 0 unverifiziert._
> _Verdichtete Fassung & Backlog-Einordnung: [../../EXPERIMENTS.md](../../EXPERIMENTS.md)._

## Executive summary

No. Across every source surveyed for this update, **not a single publicly documented, reproducible method beats a sharp Pinnacle-style *closing* line** — on either 1X2 or exact-score markets. The two newest studies that actually benchmark against Pinnacle closing both return clean negative results: a 2026 Serie A structural-model paper finds a time-fitted Dixon–Coles model receives a **logarithmic-pooling weight of exactly 0.000** against the closing price (the line already contains everything the model knows), and a reproducible public ML repo finds **negative closing-line value (CLV) against Pinnacle for every model** (RF/XGBoost/SVM). Every 2024–2026 result that *does* report profit or a scoring-rule "win" is disqualified on inspection — it beats only a ~15-book **average/soft closing line** (Wilkens 2026), an **in-play** Betfair line (AFT 2026), or **seven soft books** (Egidi 2018); is self-labeled an **upper bound**; or, in the case of modern ML (graph transformers, deep learning), **never tests against bookmaker odds at all**. For a hobby predictor blending time-weighted Dixon–Coles with Pinnacle closing and optimizing a Kicktipp 1/2/3 scheme, the practical conclusion is now empirically direct: **"match the closing line" is effectively the ceiling**, and the structural model's remaining value is as a *fallback* where closing odds are missing, not as an information add-on.

---

## Finding 1 — The only studies that properly benchmark Pinnacle *closing* both find zero edge

**Confidence: High** (one primary preprint with full proper-scoring results + one reproducible public repo; unanimous votes)

This is the core new evidence and the most direct answer to research-question parts (b) and (c).

- **Pitcan 2026, Serie A** ([arxiv 2608.11505](https://arxiv.org/html/2608.11505)) is the first 2024–2026 paper found that explicitly uses the **Pinnacle closing line** as its benchmark — "the closing price of a low-margin, high-limit book is the sharpest widely published forecast" (Bet365 opening used only as pre-2012 fallback). Its verdict on a time-fitted **Dixon–Coles** structural model over n=2,660 test matches (2019–20 to 2025–26):
  - Closing line strictly beats DC on **every proper scoring rule**: **RPS 0.1905 vs 0.1972; log-loss 0.9620 vs 0.9858; Brier 0.5717 vs 0.5862** (paired RPS diff +0.0067, 95% CI [0.0046, 0.0088], excludes 0).
  - The DC model's **logarithmic-pooling weight against the market is exactly 0.000** — a genuine boundary solution confirmed on *both* validation and test periods (log-loss monotone increasing in the weight), not an optimization artifact. A shots-on-target variant likewise got weight 0.00 against the market. Conclusion in the authors' words: *"the closing price has already absorbed both."* (claims [10], [11], [12])

- **Boui public ML repo** ([github.com/zakariae-boui/football-prediction-ml](https://github.com/zakariae-boui/football-prediction-ml)) is a reproducible, correctly-constructed CLV test: bets placed at **Bet365** odds, evaluated against **Pinnacle's closing line**, using de-vigged implied probabilities, over 6,080 PL + La Liga matches (2018/19–2025/26) with Understat xG. Result: **"Closing Line Value vs Pinnacle — Negative for all models: no edge over the sharpest market."** The degenerate/circular-CLV failure mode does not apply (Pinnacle closing is used only for post-hoc evaluation, never as a feature). (claim [21])

*Caveat:* Pitcan is a single-author, non-peer-reviewed preprint on a single league (Serie A); the repo is a single non-peer-reviewed Master's project reporting a betting-backtest-tier CLV result (no RPS/Brier/log-loss vs the closing line). But both point the same, consensus-aligned direction, and Pitcan's proper-scoring + pooling-weight design is exactly the stronger evidence tier the research question prioritizes.

---

## Finding 2 — Modern ML (graph transformers, deep learning, GBTs) does not beat the market — and usually never even tests it

**Confidence: High** (multiple primary sources; unanimous votes)

Directly answers part (a): the modern-ML class the question asks about either loses to the market or is benchmarked only against other academic models.

- **HIGFormer** ([arxiv 2507.10626](https://arxiv.org/abs/2507.10626)), a 2025 player-team **heterogeneous graph transformer** on rich WyScout event data — precisely the modern-ML archetype in scope — is benchmarked **only against other ML models** (MLP, RNN, P-Graph, T-Graph, DraftRec), **never against bookmaker/Pinnacle/closing odds**, and reports **only classification accuracy** (52.19% overall), **no RPS/log-loss/Brier and no CLV/ROI**. Its accuracy would not even exceed typical bookmaker 1X2 accuracy (~53–55%). Zero closing-line-value evidence. (claims [6], [7], [9])

- **2023 Soccer Prediction Challenge** (Yeung/Bunker et al., [Springer 10.1007/s10994-024-06608-w](https://link.springer.com/article/10.1007/s10994-024-06608-w)): the authors' best 1X2 model — a **deep-learning** architecture (Inception + Transformer encoder + MLP), with a CatBoost + pi-ratings model scoring worse — **was beaten by a bookmaker-consensus model on RPS: 0.2195 vs 0.2063 (a 6.42% margin)**. Modern ML did not beat the market. (claim [8])
  - *Scope caveats:* the benchmark is a **bookmaker consensus** (not identified as Pinnacle, closing vs opening unspecified) — a softer standard than Pinnacle closing. And the universal "ML never beats the market" is mildly over-general: the same review's 2017-challenge table shows *post-hoc* ML runs (CatBoost/TabNet) reportedly beating a soft line, but those are non-Pinnacle-closing and leakage/overfit-prone.

- **Systematic review of ML in sports betting** ([arxiv 2410.21484](https://arxiv.org/html/2410.21484v1), Oct 2024): contains **zero mentions of "closing odds," "closing line," "Pinnacle," or CLV**. Its best soccer proper-scoring numbers are reported **in isolation** against generic bookmaker-implied probabilities: RPS 0.197 (a per-shot xG model, Anzer & Bauer 2021) and Berrar ratings RPS 0.2101 — never a sharp-line head-to-head. (claim [19])

---

## Finding 3 — Every 2024–2026 "profit"/scoring-win headline fails the sharp-closing-line bar on inspection

**Confidence: High** (four primary sources; unanimous votes)

Directly answers part (b): the positive-sounding results are all against soft, average, in-play, or self-flagged benchmarks.

- **Wilkens 2026, Bundesliga** ([Journal of Sports Analytics, DOI 10.1177/22150218261416681](https://journals.sagepub.com/doi/10.1177/22150218261416681)):
  - Benchmark is the **average closing line across ~15 providers** — *"the panel typically consisting of about 15 providers"* — **Pinnacle is never isolated**. This is a soft/average closing line, not a sharp one; the top ROI figure additionally depends on **line-shopping the max across the panel**. (claim [0])
  - On proper scoring rules the model does **not** clearly beat even that average market: the **market wins Brier (0.59 vs 0.63) and per-outcome calibration**; the model wins only on aggregate **log-loss (1.25 vs 1.41)**. The paper concedes xG forecasts are "slightly less well calibrated than market odds." (claim [1])
  - Reported returns (ROI 9.5% at average odds, 14.9% at best-available, 53.85 units / 567 bets) are **explicitly an upper bound** — *"indicative of latent signal quality rather than readily realisable profits."* (claim [2])

- **AFT in-play model** ([arxiv 2605.16066](https://arxiv.org/abs/2605.16066)): benchmark is **Betfair in-play** prices and the task is **live minute-interval** forecasting — **not pre-match, not closing**. Its 4.5% ROI / 5.94 Sharpe over 17,458 bets is framed as an in-play-market inefficiency, and the model actually **underperforms the market on accuracy** (70.2% vs 70.6%); authors caveat heavily (small 140-match sample, non-executable last-traded prices). Disqualified. (claim [3])

- **Egidi/Pauli/Torelli 2018** ([arxiv 1802.08848](https://arxiv.org/pdf/1802.08848)): benchmark is **seven soft books** (Bet365, Bet&Win, Interwetten, Ladbrokes, Sportingbet, VC Bet, William Hill) from football-data.co.uk — **Pinnacle never used, "closing" never mentioned** (these are pre-closing/average snapshots). On mean-probability-of-realized-outcome the Bayesian model **loses to odds-implied probabilities in every league** (e.g., Bundesliga 0.4010 vs Shin 0.4100). Yet it claims **universal profit against every book** — a textbook **overfit/soft-book contradiction**, with the sure-loss comparison "not explicitly shown." (claims [14], [15], [16])

- **Hegarty & Whelan 2024** ([IJF, S0169207024000670](https://www.sciencedirect.com/science/article/pii/S0169207024000670)): benchmark is the **average closing 1X2 odds across surveyed soft books**, not Pinnacle. Demonstrated inefficiency (favourite–longshot bias) is in the **soft-book 1X2 market**; the **Asian Handicap market is efficient**. This is a *two-markets* comparison, **not a head-to-head that beats Pinnacle's own closing 1X2 line**. (claim [13])

---

## Finding 4 — De-vig / probability-recovery methods cap at "parity minus margin" by design

**Confidence: High** (primary source; near-unanimous)

Reinforces the ceiling for part (c).

- **MDPI Mathematics 2025** ("Domain-Driven Identification of Football Probabilities," [13(24):3976](https://www.mdpi.com/2227-7390/13/24/3976)) explicitly disclaims a market beat: *"can they be used to generate a profit against the bookmaker? The answer … is no. … the best-case scenario is achieving parity with the bookmaker's accuracy level, which still results in a negative expected value due to the non-zero margins."* Its "lowest log-loss / best calibration" headline is **only against other margin-removal heuristics** (naive, normalisation, Knowles-Woodland, Shin, etc.) applied to the *same* odds — not an independent forecast, not a closing-line beat. (Benchmark is pre-match odds from an unnamed large European book across 6,000+ leagues — necessarily aggregated, not Pinnacle closing.) (claims [4], [5])

---

## Finding 5 — The closing line absorbs late information (intra-market efficiency)

**Confidence: Medium** (single blog source, but a ~58k-match original analysis; split vote 2-1)

- **Gods of Odds** ([how-accurate-are-pinnacles-closing-odds](https://godsofodds.com/en/previews/how-accurate-are-pinnacle-s-closing-odds)): across 57,986 football matches, Pinnacle's own pre-closing prices **regress to the closing line** — "when the pre-closing to closing price ratio is 95%, we see 95% returns on average" (symmetric at 105%). Closing prices are "difficult to beat"; residual errors are "likely largely random." This is an **intra-market** efficiency demonstration (it does not test an external model), and the symmetric statistic is exactly the mechanism by which obtaining prices *better* than closing earns positive CLV — so it confirms the "closing is efficient" prior rather than proving no external method can achieve CLV. (claim [20])

*(Note: a related claim that margin-removed Pinnacle closing is perfectly calibrated in 1% bins was refuted 0-3 for source quality — see Refuted, below.)*

---

## Finding 6 — Exact-score / correct-score: no CLV evidence, and no free sharp benchmark exists

**Confidence: High** for the "no evidence" claim; the data-gap point carries over from the prior review.

Directly addresses the exact-score half of the question.

- **LLM exact-score harness** ([arxiv 2608.05030](https://arxiv.org/html/2608.05030)): a Dixon–Coles-type prior (V1) reranked by an LLM harness (V2–V4). It **never benchmarks against odds, Pinnacle, or any closing line** — all comparisons are internal (V1 vs V2–V4). Only **V1 produces proper-scoring numbers** (log-loss 0.987803, Brier 0.586970, RPS 0.209451, 1X2 accuracy 53.33%); the reranked V3/V4 outputs are **orderings, not normalized probabilities**, so they cannot be scored or compared to a market at all. V4 improved exact-score Top-1 to 14.7% (vs V1's 10.0%) but did not beat V1's native 1X2 decision. Zero CLV evidence. (claims [17], [18])
- Consistent with the prior review: **no free source of historical Pinnacle correct-score odds** was found, so a sharp-line exact-score benchmark remains empirically out of reach for a hobby setup. The strongest *reproducible* exact-score finding remains the prior review's penaltyblog result that **time-weighted Dixon–Coles beats bivariate Poisson / zero-inflated / NegBin / Weibull-copula on RPS** — a model-vs-model result, not a market beat.

---

## Practical implications for the Kicktipp hobby predictor (part c)

- **"Match the closing line" is now the empirical ceiling, not just a prior.** The Pitcan result is the closest published analog to your own setup — a time-fitted **Dixon–Coles blended (log-pooled) with Pinnacle closing** — and the optimal blend weight on the structural model is **0.000**. Do not expect a DC-plus-Pinnacle blend to add information; the closing line dominates.
- **Keep Dixon–Coles as a *fallback*, not an *add-on*.** Its realistic value is covering matches/markets where a Pinnacle closing price is missing or stale (lower leagues, exotic markets, early submission windows), and generating **exact-score distributions** the 1X2 line doesn't publish — which is where a Kicktipp 1/2/3 scheme actually earns its points.
- **The remaining lever is scheme-specific score optimization, not probability improvement.** Since you cannot out-forecast the closing line, the edge in a casual, herding pool comes from **mapping the closing-line-derived probabilities onto the Kicktipp scoring rule** (which exact scoreline maximizes expected 1/2/3 points given the field's tendencies), not from a better win/draw/loss probability. That is a decision-theory problem on top of the closing line, and it does not require beating it.
- **Ignore ROI/profit headlines as design targets.** Every profit claim in this survey collapses to soft/average/in-play odds, line-shopping, upper-bound framing, or overfitting. None survives against a sharp closing line.

---

## Caveats, weak sources, and time-sensitivity

- **Only two sources actually use a Pinnacle-closing benchmark** (Pitcan 2026; Boui repo). Both are **non-peer-reviewed** — a single-author preprint and a Master's project — and **single- or two-league** (Serie A; PL + La Liga). The finding is directionally robust and consensus-aligned (Hubáček & Šír 2023), but it rests on thin *proper-benchmark* evidence rather than a body of peer-reviewed work.
- **Benchmark dilution is the dominant failure mode.** Wilkens, Egidi, and Hegarty & Whelan all use average/soft-book lines; football-data.co.uk main columns are pre-closing snapshots for older data. Any "edge" against these does not transfer to a sharp line.
- **Metric tier matters.** The strongest evidence (Pitcan: RPS/log-loss/Brier + pooling weight) is proper-scoring/CLV; the reproducible repo is betting-backtest tier (no proper scoring rules vs closing). Treat the repo as corroboration, not standalone proof.
- **Exact-score remains data-starved.** No free historical Pinnacle correct-score odds means the exact-score ceiling is *inferred* (from 1X2 efficiency + LLM-harness null) rather than *directly tested*.
- **Preprint volatility.** Pitcan (Aug 2026) and the two arxiv LLM/AFT preprints (2026) may revise; figures cited are current as of this review (Aug 2026).
- **Refuted for transparency:** a claim that the sharp AH market is provably bias-free (1-2), that Stübinger et al. 2019's ~1.58%/match RF return demonstrates an edge (1-2, no significance test, no sharp-line test), and that margin-removed Pinnacle closing is perfectly calibrated in 1% bins (0-3, blog-quality) — all failed verification and should not be relied on.

---

## Open questions

1. **Does the Pinnacle-closing null replicate in a peer-reviewed, multi-league study?** The 0.000-pooling-weight result is Serie A only, single-author. A cross-league replication (or an EPL/Bundesliga version) with proper scoring would move this from "strongly indicated" to "established."
2. **Is there any market/regime where a structural or ML model earns *nonzero* pooling weight against Pinnacle closing?** Candidates the literature hasn't cleanly tested: lower divisions, very early lines (large pre-closing move), congested-fixture / rotation periods, or player-tracking-rich contexts where the closing line may be thinner.
3. **Can anyone construct a sharp-line *exact-score* benchmark at all?** With no free historical Pinnacle correct-score odds, the exact-score ceiling is untested directly — is there a reproducible proxy (e.g., deriving CS from Pinnacle AH + totals + 1X2) that would let a hobbyist actually measure CS-CLV?
4. **How much do Kicktipp points diverge from proper scoring at the optimum?** Since forecasting can't beat the line, the open practical question is quantifying the *scheme-optimization* gain: for a herding casual pool, how many points does closing-line-optimal scoreline selection add over naive most-likely-score, independent of probability accuracy?

---

## Anhang A — Quellen (16 gefetcht)

- [primary] <https://journals.sagepub.com/doi/10.1177/22150218261416681>
- [primary] <https://arxiv.org/abs/2605.16066>
- [blog] <https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/>
- [primary] <https://www.mdpi.com/2227-7390/13/24/3976>
- [primary] <https://arxiv.org/abs/2507.10626>
- [primary] <https://link.springer.com/article/10.1007/s10994-024-06608-w>
- [blog] <https://godsofodds.com/en/previews/how-accurate-are-pinnacle-s-closing-odds>
- [primary] <https://arxiv.org/pdf/2507.10626>
- [blog] <https://github.com/zakariae-boui/football-prediction-ml>
- [secondary] <https://arxiv.org/html/2410.21484v1>
- [primary] <https://arxiv.org/html/2608.11505>
- [primary] <https://www.sciencedirect.com/science/article/pii/S0169207024000670>
- [blog] <https://www.pinnacle.com/betting-resources/en/educational/have-pinnacles-soccer-markets-become-more-efficient/qcv2uvnqtqdw98gk>
- [primary] <https://arxiv.org/pdf/1802.08848>
- [unreliable] <https://impliedscore.com/dixon-coles-model/>
- [primary] <https://arxiv.org/html/2608.05030>

## Anhang B — Widerlegte Claims (adversarial gekillt: 3)

- „The sharp Asian Handicap closing market (dominated by Pinnacle) is empirically efficient: no favourite-longshot bias, implied probabilities are unbiased estimates of win rates, and ex ante expected loss rates accurately predict ex post loss rates - strong evidence that the sharp closing line has no exploitable bias, i.e. 'matching the closing line' is effectively the ceiling for the sharp market.“ — Vote 1-2, Quelle https://www.sciencedirect.com/science/article/pii/S0169207024000670
- „The only concrete soccer profitability evidence cited is Stubinger et al. 2019 (Random Forest, ~47,856 matches across 5 leagues) at ~1.58% return per match, reported without statistical-significance testing and without any test against a sharp closing line — so it does not demonstrate closing-line value.“ — Vote 1-2, Quelle https://arxiv.org/html/2410.21484v1
- „Margin-removed Pinnacle CLOSING prices are effectively unbiased 'true' probabilities: across a large dataset, predicted outcome probabilities (in 1% bins) match actual observed win percentages, implying the closing line is calibrated and hard to beat. This directly supports the established finding that no systematic edge over Pinnacle closing exists.“ — Vote 0-3, Quelle https://godsofodds.com/en/previews/how-accurate-are-pinnacle-s-closing-odds
