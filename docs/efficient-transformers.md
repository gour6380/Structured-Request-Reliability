# Efficient Transformers: cache cost, adaptation and reliable local inference

Three connected studies examine what an efficiency claim needs before it becomes
useful in a model or service decision: a defined resource boundary, measured task
quality and explicit failure handling. This index connects attention/KV-cache
mechanics, matched-budget LoRA/QLoRA training, and frozen-adapter regression and
serving. It is a synthesis of personal portfolio experiments, not production use
or a single benchmark across all three projects.

The attention study trained small TinyStories decoders. The adaptation study
trained Qwen2.5-1.5B-Instruct adapters on MASSIVE English requests. The reliability
project consumes independently copied artifacts from that adaptation run; it
does not import or execute the training project. Keep these tasks, environments
and denominators separate.

## Three questions and the central evidence

| Question | Measured result | Decision supported | Main limitation |
|---|---|---|---|
| How does sharing KV heads affect cache cost, quality and decoding? | MHA/GQA/MQA test perplexity **5.02663 /5.02467 /5.05913**; logical cache storage **1 /0.25 /0.125**. | KV sharing reduces the stored K/V tensors; speed still depends on the workload. | One seed, different parameter counts, one M2 Pro; long-context cases include swapping and exceed the trained context. |
| What does NF4 QLoRA change under the same adapter and token budget? | LoRA/QLoRA whole-record accuracy **72.46% /72.06%** on 2,974 requests. QLoRA's observed training allocated memory is **18.98% lower**, reserved memory **68.27% higher**, optimization time **26.29% longer**. | Track allocated memory, reservation, runtime and correctness separately. | One paired seed, concurrent A40 workers, sampled memory maxima and a fixed prompt/decoder. |
| Does a frozen adapted model preserve behavior locally, and what does it cost? | QLoRA matches **8/8 reference texts and token sequences**, but only **4/8 answers are correct**. Separate validation timing: **3.27 s p50, 5.44 s p95, 6.71 generated tokens/s** on eight inputs ×3 repetitions. | Regression agreement, semantic correctness and execution cost need separate gates. | Eight reused test cases; eight repeated validation inputs; two stopped LoRA loads; no full native equivalence or cross-model speed claim. |

### Attention and KV Cache

[Pinned technical report](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/docs/technical-report.md)
· [quality/cost table](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/results/run_001/quality_cost.csv)
· [cache-growth figure](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/results/run_001/figures/cache_growth.png)
· [decode-latency figure](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/results/run_001/figures/decode_latency.png).

Three six-layer, width-384 decoders retain eight query heads and use 8/2/1 KV
heads. Each completes 20,100 updates and 164,637,696 training targets, with final
quality measured on the same 2,358,938 test targets. This matches the token budget,
not parameter counts. Thirty condition-specific cache parity checks pass.

At batch four and 640 occupied positions, logical cache storage is **45 /11.25
/5.625 MiB**. These values count the KV tensors, not total process memory. At
prompt length 128/batch one, cached median step times are **6.291 /6.419 /6.473 ms**;
at prompt length 512/batch four, **7.443 /6.811 /6.885 ms**. MHA is fastest in the
first example and GQA in the second. The tiny one-seed quality gap does not prove
superiority or equivalence. Long-context results with machine-wide swap growth
remain qualified observations, not clean architecture speed comparisons.

### Structured Request Adaptation

[Pinned report](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/results/run_001/report.md)
· [test metrics](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/results/run_001/test_metrics.csv)
· [memory/runtime table](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/results/run_001/memory_runtime.csv)
· [memory/runtime figure](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/results/run_001/figures/memory_runtime.png).

Both arms use the same initial adapter, training order, seed 2027, rank 16,
microbatch 16, five epochs, 3,600 updates and 1,172,405 supervised tokens. Final
epoch five is the primary checkpoint for both. The two workers run concurrently
on separate NVIDIA A40 GPUs using BF16 compute and FP32 adapters/optimizer state.

Observed allocated training memory is **7.041 →5.704 GiB** from LoRA to QLoRA;
reservation is **11.973 →20.146 GiB** and optimization time **2.569 →3.245 hours**.
These memory values are sampled maxima, not guaranteed peaks; concurrency can
introduce contention. The comparison applies NF4/double quantization with ordinary
AdamW, rather than reproducing every component or large-model result of the
original QLoRA paper.

The test contract requires correct intent and the exact slot multiset as well as
valid JSON/schema. The 12-answer net LoRA advantage comes from 127 LoRA-only and
115 QLoRA-only correct responses. Of the official test inputs, 26 normalize to
text already in training or validation. Excluding them leaves **72.39% /72.01%**
accuracy on 2,948 requests. These are still inspected data, with no universal
method ranking or causal quantization-only claim.

### Structured Request Reliability

[Current technical report](../reports/technical-report.md)
· [curated results and source hashes](../reports/public/results.json)
· [quality figure](../reports/public/reference-scores.png)
· [output-free executable notebook](../notebooks/public/structured_request_reliability.ipynb)
· [setup and CLI](../README.md).

The handoff independently materializes 73 files/4.15 GiB, verifies their hashes,
re-scores the four saved CUDA prediction sets, and runs strict regression gates.
The fixed eight-case MPS QLoRA result preserves prompt tokens, continuation text
and generated token IDs. Matching a reference preserves its errors too: 8/8
schema-valid outputs coexist with 4/8 entirely correct answers.

The local CLI/API shares one frozen engine, rejects invalid inputs and reports
schema-invalid output as failure. The saved real-model ASGI check covers repeated
requests, invalid input and shutdown; it does not establish network-load capacity.
LoRA loading stopped twice at the 128 MiB machine-wide swap-growth guard before
generation. Those attempts remain failures.

The separate 1 November QLoRA benchmark uses MPS BF16 on the 16 GiB M2 Pro,
Python 3.13.11 and the frozen NF4 backbone. Loading takes **4.51 s**, the first
request **7.24 s**, then two warmups are excluded before 24 timed requests.
All measured outputs are schema-valid; semantic correctness is not scored in
this timing run. Point samples show **0.944 GiB MPS allocated /2.695 GiB driver**
memory; overlapping counters are not summed or called peaks. Swap is unchanged
at the recorded points. Disk caches were not flushed. Neither these values nor
the CUDA timing values establish a cross-device speedup.

## Inspect and reproduce deliberately

All three source repositories are published. The latest local reliability
benchmark, comparison update and this synthesis are a **local release candidate**
until committed and published; the earlier public reliability snapshot does not
contain these new results. See [release identities and boundaries](release-wrap-up.md).

For inspection, use the pinned reports and tables above; no environment or model
execution is needed. Earlier public evidence packages omit datasets/checkpoints
and cannot resume their original runs. Their archived environments are not needed
by this project. For a fresh independent reproduction, follow each pinned README:

| Study | Setup/execution entry point | Prerequisite and boundary |
|---|---|---|
| Attention/KV cache | [Pinned setup and phased notebook workflow](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/README.md#install-and-open): `python3.13 -m venv .venv`, then the pinned requirements and notebook. | Fresh data preparation and three fits are needed; public evidence is not a resumable model bundle. Preserve saved notebook outputs in a separate copy. |
| Request adaptation | [Pinned setup and run configuration](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/README.md#setup): install `requirements.txt` into a new notebook environment and choose a fresh run. | The recorded run is CUDA/Python 3.12.3 with explicit batch overrides. Hardware defaults differ; new training is not an exact Mac replay. |
| Request reliability | [Locked environment, frozen-bundle restoration and CLI](../README.md#set-up-the-project). | Exact inference requires the separately retained matching bundle; a plain public clone is insufficient. |

With this project's complete compatible bundle restored, the bounded commands are:

```sh
uv sync --locked --extra dev --extra notebook --link-mode copy
.venv/bin/request-reliability verify
.venv/bin/request-reliability compare --systems qlora --count 8 --device mps
.venv/bin/request-reliability benchmark --system qlora --device mps \
  --requests 8 --repetitions 3 --timeout 60
```

These commands start new bounded inference; cached notebook replay is a separate
inspection path. Retain the memory guards and use fresh report directories.

## Lab conclusion and scope

Treat efficiency as a decision across resource, quality and operating boundaries.
Cache savings are not application-memory savings; lower allocated training memory
is not lower reservation or faster optimization; exact backend parity is not
semantic correctness. A useful handoff records all three and preserves stopped
attempts instead of converting them into successful measurements.

The [long-form article](efficient-transformers-article.md) develops this argument.
The [release wrap-up](release-wrap-up.md) records candidate identities, verification,
an interview story and pin/Featured readiness. Full native test replay, further
LoRA loading under unchanged conditions, the native performance matrix,
peak-memory/concurrency instrumentation and prompt/decoder/multi-seed ablations
are explicitly deferred. They do not roll automatically into the next lab.
