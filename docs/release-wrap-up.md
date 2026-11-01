# Efficient Transformers release wrap-up

**Prepared:** 1 November 2026. **Scope:** local evidence/release handoff for three
connected portfolio studies. This completes the index, bounded-results synthesis
and article draft. It does not close the remaining career/review work or publish
a new GitHub release or external article.

## Release identities

Read-only GitHub checks on 1 November confirmed these default-branch commits,
no tag refs and no GitHub Releases for all three repositories. The
labels below identify the three lab stages, not tags, semantic versions or
newly created public releases.

| Lab stage | Reviewable snapshot | State |
|---|---|---|
| 1 — Attention and KV Cache | [Commit `3d39af1`](https://github.com/gour6380/Attention-KV-Cache/commit/3d39af108544a06ab30728be5673ca145672a3bf) | Published evidence and local archive verified; no changes made here. |
| 2 — Structured Request Adaptation | [Commit `d012ed4`](https://github.com/gour6380/Structured-Request-Adaptation/commit/d012ed4fb58789bcd0a8c12be34dbe500b1c1995), following initial publication `d7779d5` | Published evidence verified; source project preserved unchanged. |
| 3 — Structured Request Reliability | Published base [commit `29a8ca8`](https://github.com/gour6380/Structured-Request-Reliability/commit/29a8ca8a9465f9ea9c63e18ac416c6d92077e986), plus the current local candidate | Local candidate adds the 1 November timing/comparison update, evidence index and long-form synthesis. No new commit/tag/release is implied. |

The third candidate retains package version **0.1.0**. Its exact file bytes are
identified by a separate SHA-256 candidate inventory in the local verification
receipt; a future commit would provide its public immutable identity. No Git
history, branches, tags, repository layout or earlier project files were changed
to manufacture a three-release sequence. The logical lab is connected through
the [reviewer index](efficient-transformers.md); existing repositories stay separate.

## What the third candidate contains

- Strict full-response scoring with malformed/duplicate/unknown-field rejection,
  exact intent/slot evaluation and retained invalid outputs in denominators.
- Independently verified frozen-artifact loading, versioned model identity,
  bounded regression gates and a single-engine local CLI/API.
- A locked Python 3.13 environment, executable output-free public notebook, and
  public-safe aggregates with relative source hashes.
- Separate CUDA quality, MPS regression and validation-timing evidence, including
  the failed LoRA attempts and data-overlap audit.
- [Technical report](../reports/technical-report.md), [lab index](efficient-transformers.md)
  and [long-form article draft](efficient-transformers-article.md).

The private frozen bundle, raw requests/predictions, executed working notebook,
environment, logs and verification receipts remain excluded by `.gitignore`.
The public notebook stays in replay mode with live/install/write flags disabled.
No new model or notebook execution is part of this wrap-up.

## Verification and its date boundaries

| Evidence gate | Receipt/result | What it verifies |
|---|---|---|
| Attention software | Historical **120 passes /4 CUDA skips**, 28 September | Recorded correctness gate; not rerun for the wrap-up. |
| Adaptation software | Historical **371 passes /2 skips**, 30 September | Recorded training/evaluation gate; not rerun or relabeled as a current test. |
| Reliability software | **133 passes /38 subtests**, 1 November | Includes six public-export checks for privacy, denominators and mismatch rejection. |
| Frozen handoff | **73 files /4,451,390,755 bytes** verified | Compatible model, adapters, tokenizer, requests and provenance; not included in Git. |
| Reference-score audit | All four **2,974-response** sets and **166,544 saved score fields** checked | Saved row/aggregate scores, paired comparison, novel-input slice and shared controls. |
| MPS QLoRA regression | **8/8 text/token matches; 4/8 correct** | Bounded portability/regression result, not full quality equivalence. |
| MPS validation timing | **24 timed predictions**, eight inputs ×3 | One sequential configuration's qualified cost, not population tails or HTTP capacity. |
| Wrap-up artifact review | Links, source identities, candidate inventory, ignore/privacy checks and preservation receipt | Documentation/release preparation; software and model tests were not repeated for prose-only changes. |

The long-form draft's evidence gate is complete. Author approval and external
publication remain separate. The existing short LinkedIn posts are historical
published artifacts; their publication does not publish this article or candidate.

## Interview story: preserving a result without preserving a mistaken claim

> I took trained structured-request adapters into an independent local inference
> project on a 16 GiB M2 Pro. I froze the prompt, tokenizer, ontology, weights and
> decoding configuration, verified the copied bundle, and used exact prompt and
> generated-token checks before timing anything.
>
> QLoRA matched all eight selected CUDA outputs on MPS, yet only four answers were
> entirely correct. That made the acceptance contract visible: reference agreement,
> schema validity and semantic correctness were separate measurements. LoRA
> loading stopped twice at a machine-wide swap-growth guard; I kept those failed
> attempts instead of weakening the guard to obtain a latency number.
>
> I then measured one viable QLoRA configuration on eight fixed validation inputs,
> three repetitions, two excluded warmups and synchronized timing: 3.27 seconds
> median and 5.44 seconds p95. I reported memory as overlapping point samples and
> kept the reused test and small-input limits explicit. The engineering outcome
> was an inspectable handoff and failure-aware local interface, not a claim of
> universal equivalence or production-quality request parsing.

## Pin and Featured readiness decision

**GitHub:** Attention and KV Cache is ready as the mechanism-focused entry point.
Structured Request Adaptation is ready as the controlled-training comparison.
For the combined story, prefer the reliability repository once this candidate's
index and timing/report update are published at a reviewed immutable commit.
Until then, its live commit supports the earlier scoped result only.

**LinkedIn Featured:** the existing adaptation or reliability post can introduce
the work, with their bounded claims retained. Add the long-form article only after
author review and a live publication link exists. This is a readiness decision;
no profile edit or external publication was performed.

## Closed optional work and remaining actions

The following are deferred for this result: full 2,974-request MPS replay; native
base/NF4/LoRA performance comparisons; further LoRA loading under unchanged memory
conditions; peak-memory/concurrent-serving experiments; tuned prompts, alternate
decoders and multi-seed superiority/equivalence studies. No claim depends on
treating any of these as completed. Reopening one requires a new bounded protocol
and a reason to change the current decision.

The next concrete action is author review of the candidate and article, followed
by any separately authorized publication. The program's application/profile block
and weekly retrospective remain open; unreported hours and outcomes are not
inferred. This handoff does not start another experiment or lab.
