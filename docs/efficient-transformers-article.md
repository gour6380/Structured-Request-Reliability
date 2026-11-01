# What an efficient Transformer result needs to prove

**Gourav Karwasara · Independent portfolio studies · Draft, 1 November 2026**

Eight matching outputs can still contain four wrong answers.

That was the most useful result from moving a frozen QLoRA model from CUDA to
Apple Silicon. The local model reproduced all eight reference continuations,
including their generated token IDs. Every response satisfied the JSON schema.
Only four responses had the correct intent and exact slot values.

The handoff worked on those requests. The model's task errors survived it too.

This distinction connects three studies I built around Transformer efficiency:
attention and KV caching, matched LoRA/QLoRA adaptation, and frozen-model
evaluation with local serving. Each produced a useful result, but each answered
a different question. Smaller cache tensors, lower observed training allocation,
matching outputs and lower request latency are separate observations. Turning
them into one claim about an “efficient model” would hide the boundaries that
make the evidence useful.

My working rule is to require a measurement boundary, a quality contract and an
account of failures together. These studies support that rule through concrete
counterexamples rather than a universal ranking of architectures or runtimes.

## Cache savings have a precise meaning

The attention study compared small decoders trained from scratch on TinyStories.
The six-layer models shared width 384, eight query heads, tokenizer, data and an
equal training-token budget. MHA used eight KV heads, GQA two and MQA one. This
applies the head-sharing idea described in the [GQA paper](https://arxiv.org/abs/2305.13245);
it does not reproduce that paper's checkpoint-uptraining experiment.

Each variant completed three epochs, 20,100 optimizer updates and 164,637,696
target tokens. Final test perplexities were **5.02663 MHA, 5.02467 GQA and 5.05913
MQA**. GQA had nearly matching measured quality in this run. Its tiny numerical
advantage does not establish a quality improvement across training seeds.
Reducing KV projections also changed parameter counts, so the comparison holds
the token budget constant rather than matching parameters.

The cache calculation was less ambiguous. Logical storage is:

```text
2 × layers × batch × KV heads × retained positions × head dimension × bytes/value
```

At batch four and 640 retained positions, FP32 KV tensors occupied **45 MiB for
MHA, 11.25 MiB for GQA and 5.625 MiB for MQA**. The ratios were exactly
**1, 0.25 and 0.125**. That describes the stored keys and values, not total process
memory. Weights, intermediate tensors and runtime allocations remain outside it.

Latency supplied the counterexample. With a 512-token prompt, batch four and 128
append steps, cached median step times were **7.443, 6.811 and 6.885 milliseconds**
for MHA, GQA and MQA. The smallest cache did not produce the fastest step. At
prompt length 128 and batch one, MHA was fastest at **6.291 milliseconds**, versus
6.419 for GQA and 6.473 for MQA.

Caching itself delivered a much larger benefit: at prompt 512/batch four, the
uncached-to-cached median ratios were roughly eightfold. Those timings used
equivalent fixed token values and 128 append forwards, with accelerator
synchronization and separate prefill timing. Thirty condition-specific parity
checks passed before valid timing.

This was explicit attention computation rather than a competition among the
fastest available kernels. Longer workloads also exceeded the trained context
and encountered changing system swap. They cannot establish long-context
language quality or a clean hardware ranking. The [pinned attention report](https://github.com/gour6380/Attention-KV-Cache/blob/3d39af108544a06ab30728be5673ca145672a3bf/docs/technical-report.md)
keeps those conditions beside the results.

## Adaptation changes the quality contract and the cost

The next study used Qwen2.5-1.5B-Instruct to convert MASSIVE English requests into
an intent and slot list. [LoRA](https://arxiv.org/abs/2106.09685) supplies trainable
low-rank updates while freezing the backbone. [QLoRA](https://arxiv.org/abs/2305.14314)
combines such updates with a frozen four-bit backbone. My quantized arm used NF4
and double quantization, with ordinary AdamW rather than the paper's paged
optimizer. This was a controlled application, not a reproduction of the papers'
large-model benchmarks.

Both arms used the same initial adapter, training order, seed, rank, target
modules and five-epoch budget: **3,600 updates and 1,172,405 supervised tokens**.
They ran concurrently on two NVIDIA A40 GPUs, one worker per arm. The primary
comparison used final epoch-five checkpoints for both methods.

The scorer required the entire response to be raw JSON with the correct intent
and exact multiset of slot type/value pairs. Extra prose, code fences, unknown
labels and incorrect value types failed. Case and repeated slots mattered;
key order did not. Invalid outputs stayed in the denominator.

| System | Entire response correct, 2,974 requests | Schema valid |
|---|---:|---:|
| Dense base | 5.68% | 59.28% |
| NF4 base | 0.27% | 3.19% |
| LoRA | 72.46% | 99.56% |
| QLoRA | 72.06% | 99.66% |

The 12-answer gap between the adapted systems conceals substantial disagreement:
2,028 answers were correct in both, 127 only in LoRA, 115 only in QLoRA and 704
in neither. One paired seed does not establish superiority or equivalence.

The unusually poor NF4 baseline also needs interpretation. **2,745 responses
began with Markdown fences**, accounting for most of its 2,752 strict JSON
failures. The contract deliberately rejects those wrappers. That is evidence
about format compliance under this prompt and implementation; it does not show
that four-bit quantization erased semantic knowledge.

The official splits were retained with duplicates disclosed. Excluding 26 test
utterances whose normalized text also occurred in training or validation left
2,948 requests and similar accuracy: **72.39% LoRA and 72.01% QLoRA**. This check
does not remove semantic near-duplicates, audit pretraining exposure or make an
already inspected test set untouched.

Memory and time moved in different directions:

| Training measurement | LoRA | QLoRA |
|---|---:|---:|
| Observed CUDA allocated memory | 7.041 GiB | 5.704 GiB |
| Observed CUDA reserved memory | 11.973 GiB | 20.146 GiB |
| Synchronized optimization time | 2.569 h | 3.245 h |

QLoRA's observed allocation was **18.98% lower**, while reservation was **68.27%
higher** and optimization took **26.29% longer**. Allocation and reservation are
overlapping allocator measures; they cannot be added. Memory values are maximum
observed samples, not guaranteed peaks. Concurrent workers also share host
resources, qualifying the timing comparison.

The [pinned adaptation report](https://github.com/gour6380/Structured-Request-Adaptation/blob/d012ed4fb58789bcd0a8c12be34dbe500b1c1995/results/run_001/report.md)
also discloses a legacy telemetry aggregation defect. Cumulative sample metadata
could not be recovered and remains unavailable. Reporting preserved that limit
rather than inventing a complete historical memory trace.

These results support adaptation gains over this fixed prompt and decoder.
They do not compare against tuned prompting or constrained decoding. LoRA versus
QLoRA also changes training arithmetic and learned adapters together; it does
not isolate quantizing one common trained adapter at inference.

## A frozen handoff preserves behavior, including errors

For local inference, I created an independent bundle containing the matching
backbones, final adapters, tokenizer, prompt, ontology and reference outputs.
All **73 files, totaling 4.15 GiB**, passed hash verification. Runtime uses that
bundle rather than importing the training project.

The first eight test requests were fixed by file order. The regression gate
allowed no loss of whole-record correctness or schema validity. Prompt tokens,
continuation text and generated token IDs were checked separately. QLoRA matched
all eight CUDA references on MPS, with **4/8 entirely correct and 8/8 schema-valid**
in both environments. This is a portability observation on a reused prefix.
CUDA batch 32 became MPS batch one, so device, batching and kernels changed
together.

LoRA loading stopped twice when system-wide swap grew beyond the 128 MiB guard.
Both attempts completed zero predictions. The failed arm remains failed; the
passing QLoRA arm cannot turn the combined comparison into an overall pass.
Machine-wide swap also cannot attribute all growth to the model process.

The local service carries this distinction into its interface. Invalid input
returns 422, a busy model 503, a deadline overrun 504 and schema-invalid model
output 502. Cancellation retains model ownership until native work exits.
Real-model checks covered repeated responses, invalid input and shutdown in
process. They did not establish concurrent serving capacity.

The practical benefit of freezing was an inspectable boundary: a known artifact
can be loaded, its behavior compared and its failures retained. Matching text
answers whether the handoff reproduced behavior. The semantic scorer answers
whether that behavior satisfied the task.

## Local timing needs its own denominator

A separate QLoRA benchmark ran on the 16 GiB M2 Pro in the restored native Python
3.13.11 environment. It used eight validation inputs, one separately timed first
request, two excluded warmups and three measured repetitions. That produced
24 measurements from eight distinct inputs.

| Measurement | Recorded result |
|---|---:|
| Model loading | 4.5144 s |
| First request | 7.2373 s |
| Warm median / p95 | 3.2676 /5.4403 s |
| Generated-token throughput | 6.7104 tokens/s |
| Measured schema-valid responses | 24/24 |

The request timer includes tokenization, transfer, generation, synchronization,
decoding, schema validation and guard overhead. It excludes loading and the
subsequent memory snapshot. Throughput uses summed timed request durations and
includes terminal tokens in its token count. The p95 is the 23rd sorted duration
of 24, with little evidence about a population tail. This timing run computes
no semantic accuracy score.

MPS driver memory was **2.695 GiB at the recorded request points**. Those samples
are not peaks. System swap matched preflight at every measured point; that does
not mean swap was unused or continuously monitored. Disk caches were not flushed,
so loading and first-request values are not storage-cold startup measurements.
Different inputs, batches and timer boundaries also prevent a CUDA speedup claim.

## Reproduce the bounded question

The current [technical report](../reports/technical-report.md) and
[aggregate evidence](../reports/public/results.json) contain the local candidate's
new timing. This article is an unpublished draft; those additions are not claimed
to exist in the earlier published source snapshot. The [lab index](efficient-transformers.md)
and [release wrap-up](release-wrap-up.md) distinguish published commits from the
current candidate.

From the reliability project root, restore its isolated environment and complete
private frozen bundle before executing a new bounded run:

```sh
uv sync --locked --extra dev --extra notebook --link-mode copy
.venv/bin/request-reliability verify
.venv/bin/request-reliability benchmark --system qlora --device mps \
  --requests 8 --repetitions 3 --timeout 60
```

A public clone contains code and selected evidence, not the trained weights,
private data bundle or raw local runs. A fresh base download cannot substitute
for those adapters and reference outputs. The command creates a new result
directory and retains the original memory guards.

Full native replay, native baseline timing, another same-condition LoRA load,
peak-memory instrumentation and prompt/decoder or multi-seed ablations remain
deferred. They are useful possible experiments, not evidence already obtained.

Across these studies, the most durable engineering result is knowing which
question each measurement answers. Cache storage can shrink without a latency
win. Training allocation can fall while reservation and runtime rise. A runtime
can faithfully reproduce a wrong answer. Keeping the measurement boundary,
quality contract and stopped runs together makes those outcomes useful design
evidence.
