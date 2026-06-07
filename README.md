# Redrob Ranker — Intelligent Candidate Discovery & Ranking

Ranks 100,000 candidates for the **Senior AI Engineer (Founding Team)** job
description and returns a top-100 a recruiter can trust — by reasoning about
what the role *means*, not by counting keywords.

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs in **~3 seconds** on the full 100K pool, on **CPU**, with the **network
off**, using the **Python standard library only** — no model downloads, no API
calls, nothing to install.  The pipeline parallelises across CPU cores
(`multiprocessing`) and a conservative cheap pre-filter culls the ~76 % of the
pool with no AI signal anywhere, so the expensive text scan only runs on
candidates that can plausibly reach the top 100.

---

## The core idea

The JD is explicit that the naive answer is a trap:

> *"The right answer is not 'find candidates whose skills section contains the
> most AI keywords.' A candidate who has all the AI keywords but whose title is
> 'Marketing Manager' is not a fit. A Tier-5 candidate may not use the words
> 'RAG' or 'Pinecone', but if their career history shows they built a
> recommendation system at a product company, they're a fit."*

So the system is built around three principles:

1. **Read the work, not the labels.** Domain fit is mined from the free-text
   *career-history descriptions* and summary, not from the (easily stuffed)
   skills array.
2. **Availability is part of fit.** A perfect-on-paper candidate who's been
   inactive for six months with a 5% recruiter-response rate is, for hiring,
   not actually available — and is down-weighted.
3. **Impossible profiles are removed.** ~80 honeypots with internally
   contradictory profiles are forced to the bottom by a consistency gate
   (honeypot rate in our top 100: **0%**).

## How it scores

```
final = base_fit × disqualifier_mult × behavioural_mult × honeypot_mult
```

**`base_fit`** is a transparent weighted sum of seven bounded [0,1] components
(`redrob_ranker/scoring.py`):

| Component | Weight | What it captures |
|-----------|:------:|------------------|
| role / title | 0.22 | Is this actually an AI/ML/search/data engineer? (`A/B/C/D` title tiers; a current non-core title can be redeemed by a clearly core *past* role; a junior-titled candidate is multiplied by 0.55) |
| domain match | 0.32 | Retrieval / ranking / recsys / NLP evidence **mined from career-history text**, IR-weighted |
| experience | 0.14 | Peak at 6–8 yrs, plateau 5–9, steep decay outside so 3-yr / 15-yr outliers can't sneak in on availability alone |
| company | 0.12 | Product-company exposure vs. whole-career IT-services/consulting |
| skills | 0.10 | AI skills **corroborated** by endorsements + months-used + assessment scores (anti-stuffing) |
| education | 0.04 | Light institution-tier nudge (JD weights culture/skills over pedigree) |
| location | 0.06 | Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR/Bangalore welcome; relocation considered |

The role share is intentionally lower than a naive "role first" weighting
because the JD explicitly allows a Tier-B candidate with strong career-history
domain evidence (a "Tier-5 recovery") to compete.

**Three multiplicative gates** encode the JD's hard rules:

- **Disqualifier** (`×0.40–0.75`): entire-career consulting, vision/speech/
  robotics-only with no NLP/IR, research-only without production, recent-
  framework-only ("LangChain demos"), title-chasing, manager/architect with no
  recent hands-on signal.
- **Behavioural availability** (`×0.70–1.18`): last-active recency, recruiter
  response rate, open-to-work, interview-completion, notice period, recruiter
  saves.
- **Honeypot/consistency** (`×0.01` when impossible): see below.

The behavioural multiplier is intentionally allowed to exceed 1.0 so it can
*separate* otherwise-tied strong candidates — important because **NDCG@10 is 50%
of the score**. `rank.py` normalises the final internal score to [0,1] for the
`score` column.

## Honeypot detection (`redrob_ranker/honeypots.py`)

No IDs are hard-coded. We read each profile for genuine impossibilities — the
same inspection a careful recruiter would do:

- a *current* role claiming more months than have elapsed since its start date;
- a role that ends before it starts;
- career tenures summing far above stated years of experience;
- "expert/advanced" proficiency in skills used **0 months**;
- assessment scores for skills that aren't on the profile;
- experience that predates the start of education.

Genuine impossibilities force the candidate to the bottom; softer overclaims
apply a graded penalty.

## Reasoning (`redrob_ranker/reasoning.py`)

Each row gets specific, non-templated, honest reasoning built **only** from
facts read off the candidate — citing real matched phrases and real signal
values, never inventing skills, and switching tone with the score (so a
low-ranked row reads critically, a high-ranked row reads positively). This is
what Stage-4 manual review checks for.

Example output:

```
CAND_0000031,1,1.0000,"Recommendation Systems Engineer with 6 yrs; profile shows
  hands-on recommendation system, learning-to-rank — the retrieval/ranking work the
  role centres on; Hyderabad-based, responsive and recently active, open to work."
```

---

## Repository layout

```
rank.py                              # entrypoint — multiprocessing + pre-filter (stdlib only)
redrob_ranker/
  concepts.py                        # the JD knowledge model (concepts, titles, firms, cities)
  features.py                        # cheap per-candidate feature extraction
  scoring.py                         # 7 components + 3 multipliers + composite
  honeypots.py                       # internal-consistency / honeypot gate
  reasoning.py                       # deterministic, grounded, non-templated reasoning
  prefilter.py                       # cheap "can this candidate plausibly reach top-100?" cut
  io_utils.py                        # streaming .jsonl/.jsonl.gz loader
eval/
  metrics.py                         # NDCG@10/@50, MAP, P@10, composite
  run_eval.py                        # score both labelled sets (career-history + headline-only)
  (seed_labels.jsonl)                # generated by make_seed_labels.py — git-ignored
  (seed_labels_headline.jsonl)       # generated by make_seed_labels_headline.py — git-ignored
tools/
  make_seed_labels.py                # career-history rubric (same lens as the ranker — regression)
  make_seed_labels_headline.py       # headline-only rubric (different lens — cross-validation)
sandbox/app.py                       # sample-only Streamlit sandbox (mandatory demo)
tests/test_basic.py                  # 8 fast tests (honeypot, stuffer, off-domain, format…)
validate_submission.py               # the organisers' format validator (vendored)
submission_metadata.yaml             # portal metadata
```

## Quickstart

```bash
# 1. Reproduce the submission (no install needed)
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 2. Validate the format
python3 validate_submission.py submission.csv      # -> "Submission is valid."

# 3. Run the tests
python3 tests/test_basic.py                        # -> 8/8 passed

# 4. Try the sandbox on the synthetic sample
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

## Evaluating without a leaderboard

The competition has **no live leaderboard**, so we measure quality offline
against **two independent label sets** — a deliberately stricter test than a
single rubric, since the first rubric and the production ranker share their
text-extraction lens:

```bash
# career-history rubric (same lens as the ranker — regression guard)
python3 tools/make_seed_labels.py --candidates ./candidates.jsonl \
    --out eval/seed_labels.jsonl

# headline-only rubric (deliberately different lens — cross-validation)
python3 tools/make_seed_labels_headline.py --candidates ./candidates.jsonl \
    --out eval/seed_labels_headline.jsonl

python3 eval/run_eval.py --candidates ./candidates.jsonl
```

`run_eval.py` reports NDCG@10/@50, MAP, P@10 and the composite on both label
sets, plus a **honeypot-leakage** check on the predicted top-10.  Holding up on
the cross-lens rubric is the harder, more honest result.

The seven component weights are **hand-set priors** defined in
`redrob_ranker/scoring.py`. They are not learned and require no training step —
edit them there if you want to re-balance the model.

## Compute compliance (Stage 3)

| Constraint | Limit | This system |
|------------|-------|-------------|
| Runtime | ≤ 5 min | ~3 s for 100K (8 workers) |
| Memory | ≤ 16 GB | ~63 MB aggregate / ~8 MB per worker |
| Compute | CPU only | CPU only |
| Network | off during ranking | no network; stdlib only |
| LLM calls during ranking | none | none |

Pre-computation is not required — there are no embeddings or model weights to
build.

**No AI model is used at runtime.** The ranker is a deterministic, rule-based +
text-matching scoring function on the Python standard library. It does not call,
load, or depend on any machine-learning model, LLM, or external service at any
point in the ranking pipeline. (Per the hackathon rules, AI *development* tools
are permitted and declared honestly in `submission_metadata.yaml`.)

## Notes on the hackathon terms

- **The dataset is never committed.** `candidates.jsonl` is git-ignored
  (confidential per the ToS, and 465 MB). Only the path is referenced.
- **No third-party IP.** The ranker is original work on the Python standard
  library; the only optional dependency (the sandbox demo) is permissively
  licensed.
- **No LLM/API/GPU at ranking time** — see the compliance note above.

## Design limitations & next steps

- The domain signal is a curated concept model over text. A dense
  bi-encoder (e.g. BGE/E5) could be **pre-computed offline** and loaded at rank
  time as an additional signal for harder Tier-5 paraphrases — added as a drop-in
  component without changing the compute profile of the ranking step.
- Seed labels are a heuristic rubric; human review of the labels would tighten
  both the regression and cross-lens evaluation signals.
- The cheap pre-filter is intentionally conservative (it lets ~24 % of the pool
  through to full scoring). A profile-completeness or recency-weighted upper
  bound could cut another factor without quality loss.
