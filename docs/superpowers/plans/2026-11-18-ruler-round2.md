# RULER 16K Bounded-Swap Round 2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Independent dataset and GPU-backend modules are delegated under superpowers:dispatching-parallel-agents; integrate and verify all interfaces before runs.

**Goal:** Execute the user-approved steps 1–4: old-Trace quality replay, official RULER BF16/KV4 baselines, independently calibrated/tested Oracle closed-loop quality, and complete selector cost accounting.

**Architecture:** A new CPU reference selector isolates online work from audits. An independent batched Torch selector and grouped FP32 QK/AV attention backend enable GPU closed-loop runs. Freeze official RULER v1 generator/corpus, split seeds, selection rules and inference policy in versioned manifests. Stream shadow metrics instead of exporting huge Q/K/V traces.

**Tech Stack:** Existing project Python 3.12 / NumPy / PyTorch 2.5.1+cu121 / Transformers 4.51.3; one RTX 3090. Additional data-generator dependencies only when required, pinned separately.

**Spec:** `docs/2026-11-18-next-experiment-assessment.md`; user explicitly approved continuing its four execution priorities.

## Global Constraints

- New branch in the current workstation checkout; preserve existing untracked analysis and all old results. No repeated approval or separate design round: the user approved the prior assessment.
- Page=32, sink=1 page, local=128 tokens, fixed B=floor(J×fraction), deduplicated physical pages; deterministic score/Page-ID ties.
- A1, A3, A3-B(b=8), gate-A3, one-swap main δ=.01; sensitivities δ=0/.02 with one swap and δ=.01 with two swaps. Separate α=1/.95/.90 upper-bound diagnostics; evaluate original targets.
- Main swap candidates: last 4 selected nonprotected / first 8 unselected in A1 order. Strictly positive original-deficit reduction; initial-good-head floor max(p,C_A1−δ), fixed relative to initial A1 across both rounds. Tie: gain desc, in-ID asc, out-ID asc.
- Replay all nine old (p,budget) cells; focused costs at (.8,.20),(.9,.10).
- Official RULER v1 pinned source and exact corpus checksums, four subtask definitions (three Multi-Key and Multi-Value); 8 engineering + 32 calibration + 128 test examples per task, independent seeds/instances. 16K includes reserved output and final chat template.
- Greedy generation, EOS-aware, at most 128 new tokens INCLUDING the prefill-produced first token; full Dense Prefill for every method. All 36 layers / all 8 KV groups use the selected decode policy. No L0 special case.
- BF16 and KV4 numeric-reference baselines use identical grouped FP32 decode attention and SDPA prefill; only cache quantization differs. No packed-KV4 or system-speed claim.
- Primary formal configuration (.8,.20); (.9,.10) is a prespecified stress configuration, not a second chance to choose a winner.
- Calibration shadows sample layers [0,7,14,21,28,35], decode steps [0,1,3,7,15,31,63,127] when reached. Same Dense KV4 trajectory and shared V for comparisons; no original-QKV export required.
- Select one bounded candidate on calibration only. Eligibility: mean prompt absolute P99 at least .5% better than A1 and mean prompt relative P99 no more than 1% worse; choose lowest mean absolute P99, ties prefer fewer swaps then smaller δ. If none eligible, freeze main δ=.01/one-swap as a prespecified falsification candidate, not a claimed winner.
- Freeze selection before test labels/metrics inspection. Test BF16, dense KV4, A1 and frozen swap at both focus settings; reuse identical KV4 prefill states safely, never mutate a shared prefix.
- Two prespecified primary-configuration utility routes, with 97.5% paired prompt-cluster bootstrap intervals: (a) complete-answer accuracy improves, absolute P99 noninferiority within +1%; or (b) mean prompt absolute P99 improves at least 1% with interval below zero, complete-answer noninferiority within −1 percentage point. Also require focused CPU online-selector median overhead ≤25% over A1. Report uncertainty and all failed conditions; do not relax thresholds after results.
- Official fractional RULER score AND complete-answer correctness, all-gold-span/page recall, absolute/relative P95/P99, raw-target deficit/P5, pages/gates/swaps/candidate counts. Prompt-level paired statistics, task-stratified bootstrap with fixed seed, no head/step pseudo-replication.
- Exact Oracle scoring remains a quality reference. Count/timer labels distinguish common scoring, online selection, audit, NumPy/CPU comparator reference and Torch executed work.

## Review Focus

1. Swaps preserve B, protection and initial A1 guards over two rounds; deterministic ties and no-op cases.
2. GPU and CPU selection agree, including all-protected, p=1, b=1 controls and zero mass.
3. Official data prefixes/values survive the chat template; all gold values mapped after final tokenization; no split overlap or output-budget overflow.
4. EOS/new-token counting, full K=J selected degeneration, BF16/KV4 backend parity and shared-prefill immutability.
5. Online timing excludes audits; calibration freeze predates test evaluation; incomplete/resumed runs never silently mix configurations.

### Task 1: CPU reference, old replay and costs (root)

**Files:** `gqa_probe/online_selection.py`, `gqa_probe/replay_round2.py`, `tests/test_online_selection.py`; new result directories.
**Interfaces:** `select_online(mass[J,H],protected,budget,target,algorithm='A1',delta=.01,max_swaps=1,batch_size=8,alpha=1,count_comparisons=False) -> result(order,coverage,metrics)`.

- [ ] Add failing tests for hand-derived swap, initial-anchor guards, gate/no-op, physical budget, α=1 and old A1/A3/b8 equivalence.
- [ ] Implement reference decisions and explicit arithmetic/ranking/candidate/update counters, with audits outside measured online time.
- [ ] Replay old NPZ and export per-prompt/full-grid summaries. Benchmark both focus settings with warmup, randomized method order and repeated timings; include candidate construction and coverage updates.

### Task 2: Official dataset (data agent)

**Files:** `gqa_probe/ruler_data.py`, `scripts/prepare_ruler.py`, `tests/test_ruler_data.py`, `requirements-ruler.txt`, `configs/ruler_round2.json`.
**Interfaces:** One JSONL for each split. Row keys: `sample_id`, `task`, `split`, `input_ids` (final chat-token IDs), `references` (list[str]), `gold_spans` (all relevant token [start,end) intervals), `input_tokens`, `max_new_tokens=128`, `source_metadata`.

- [ ] Test scoring/length/split/span mapping with literals before implementation.
- [ ] Pin official generator and corpus, install only required data dependencies, generate 8/32/128 per task with disjoint seeds, preserve exact inputs and hashes.
- [ ] Export provenance manifest and validate official task score vs complete-answer metric; do not run GPU inference or change other modules.

### Task 3: Torch selection and attention (GPU agent)

**Files:** `gqa_probe/torch_selection.py`, `gqa_probe/oracle_attention.py`, `tests/test_torch_selection.py`, `tests/test_oracle_attention.py`.
**Interfaces:** `select_torch(mass[G,J,H],protected,budget,target,algorithm='A1'|'SWAP',delta=.01,max_swaps=1) -> result(mask[G,J],coverage[G,H],gate_hit[G],swaps[G],candidate_pairs[G],metrics)`.
`OracleAttention(algorithm='dense'|'A1'|'SWAP', budget_fraction=.2,target=.8,delta=.01,max_swaps=1,shadow_specs=None,sample_layers=None,sample_steps=None,callback=None,gold_spans=None)` context manager; `.step` set by runner, prefill −1. Callback records per-group shadows with layer/step/group/spec, head errors/coverage, gold recall and counters.

- [ ] Test Torch fixture decisions and CPU consistency, small Qwen backend parity, EOS handled by root runner, full-mask degeneration and independent cache prefix references.
- [ ] Group QK/AV by KV head (no expanded history × GQA duplication), FP32 scores/Softmax/AV, BF16 output; SDPA for multi-token prefill. Oracle mass normalized over full valid history.
- [ ] Stable selected-token softmax, full original gold spans, actual/ideal cost labels, batched active gating; no per-token bulk history transfer to CPU.

### Task 4: Calibrated experiment and paired reporting (root)

**Files:** `gqa_probe/ruler_experiment.py`, `gqa_probe/paired_statistics.py`, tests, configs and result reports.

- [ ] Test EOS/128-token counting, safe shallow cache cloning (DynamicCache appends never mutate shared prefix), official complete/fractional metrics, grouped bootstrap and frozen-config checks.
- [ ] Engineering BF16/KV4 first; resolve quantization/backend defects before formal runs.
- [ ] Calibration shadows, freeze candidate, test all methods and both focus budgets on the same 512 held-out examples. Stream checksummed progress and allow exact-configuration resume.
- [ ] Run full tests, independent review, audit complete counts and input hashes; generate full main/stress tables and Stage decision. Keep hardware/metadata unvalidated.

## Execution ledger

`docs/round2-progress.md` records tasks, rulings, exact commands and completed run counts. Data and GPU agents have disjoint file ownership; root installs no packages concurrently with data agent.
