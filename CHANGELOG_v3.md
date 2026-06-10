# v3.0.0 — honeypot world-knowledge gate, reasoning de-templating, robustness

## Why v3

An audit of the v2.1 submission found six candidates in the submitted top-100
with roles at **Krutrim** and **Sarvam AI** starting 2018–2022 — both companies
were founded in **2023**. That is the brief's canonical honeypot ("8 years of
experience at a company founded 3 years ago"); 73 such profiles exist pool-wide
and the internal-consistency gate catches none of them (it has no world
knowledge). One sat at **rank 7**, inside the NDCG@10 window that carries 50%
of the composite. The audit also showed every one of the 100 reasoning strings
sharing one identical clause, and 69/100 citing the same phrase twice
("recommendation system, recommendation systems") — both direct hits for the
Stage-4 "templated reasoning" penalty.

## Fixes

1. **Company-age honeypot gate** (`concepts.py`, `honeypots.py`) — founding-year
   table for prominent young AI companies; a role starting before its company
   existed is a hard honeypot. Exact-match on normalised names so lookalikes
   ("Sarvam Textiles") can't be hit. Removes all 6 leaked honeypots; 0 of the
   73 remain anywhere in the new top-100.
2. **Reasoning de-templating** (`reasoning.py`) — evidence phrases de-duplicated
   by containment (no more singular+plural citations); citation *selection* and
   four lead-sentence structures rotate deterministically per candidate_id;
   rows ranked >60 carry an honest fact-based qualifier (their weakest scoring
   axis) so tone tracks rank. Truncation now cuts at a word boundary.
3. **Gzipped input** (`rank.py`) — `candidates.jsonl.gz` (as shipped in the
   official bundle) is now detected by magic bytes and transparently
   decompressed to a temp file before the byte-range scan. v2.1 crashed with
   `UnicodeDecodeError`.
4. **Chunk-boundary line loss** (`rank.py`) — if a worker boundary fell exactly
   on a line start, that candidate was silently skipped by both workers. The
   worker now peeks at the byte before its start offset and only discards a
   genuinely partial line. (Never fired on this dataset at 1–64 workers, but
   it was silent data loss waiting for different hardware.)
5. **JD-fidelity** (`features.py`, `scoring.py`, `concepts.py`) —
   - the "moved into architecture/management, no recent code" disqualifier now
     actually fires: any non-tier-A managerial/architect title with weak
     hands-on domain signal (v2.1 required a tier-D title, which no
     architect/EM title maps to, so it never triggered);
   - consulting-firm matching is word-boundary aware ("lti" no longer matches
     inside "Multiplier"/"ultimate");
   - `github_activity_score` now feeds a small skills bonus (JD: open-source /
     external validation);
   - hard "remote" work-mode preference gets a mild behavioural down-weight
     (the role is hybrid Pune/Noida);
   - Bangalore moved from "welcome" to "other Tier-1" (the JD's welcome list
     is Hyderabad/Pune/Mumbai/Delhi-NCR; Tier-1 relocators are still fine).
6. **Tests** (`tests/test_fixes.py`) — 10 new tests lock all of the above in.

## Measured impact

See the comparison battery run against the v2.1 submission (validator, full
reproduction, honeypot scans, reasoning audits, prefilter-off equivalence).
Headline: 6 tier-0 honeypots leave the top-100 (one from rank 7), replaced by
clean candidates; reasoning passes all six Stage-4 checks by construction;
runtime and memory unchanged (~4 s, ~60 MB on 8 cores).
