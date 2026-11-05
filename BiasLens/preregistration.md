# Study design

Hypotheses and metrics are fixed here before the experiment is run, so the analysis is
decided in advance rather than chosen to fit the results.

## Models

Three models served on Groq (see `config.yaml`), decoded deterministically (temperature 0).

## Hypotheses

**H1 — Framing effect.** When the same expected-value choice is framed as a gain versus a
loss, a rational agent is indifferent. We expect models to prefer the sure option more
under gain framing than loss framing.
- Metric: `framing_effect = P(sure | gain) − P(sure | loss)`, averaged over problems.
- Detectable if the 95% bootstrap confidence interval excludes 0.

**H2 — Overconfidence.** On factual multiple-choice questions, we expect stated confidence
to exceed accuracy.
- Metric: `overconfidence_gap = mean(confidence) − accuracy`. Supporting: ECE, Brier score.
- Detectable if the 95% bootstrap confidence interval excludes 0.

**H3 — Gender bias.** In ambiguous contexts where the correct answer is "unknown", we
expect models to sometimes commit to a gendered answer instead.
- Metric: `ambiguous_overcommitment` — the rate of choosing a concrete answer when the
  context does not support one.
- Detectable if the 95% bootstrap confidence interval excludes 0.

## Analysis

- Fixed dataset built from versioned templates and seeded benchmark samples.
- Deterministic decoding; `n_reps` repetitions per item.
- Confidence intervals by non-parametric bootstrap with a fixed seed.
- Every raw response is stored; only un-parseable outputs are excluded, and their count is
  reported.
