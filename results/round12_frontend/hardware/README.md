# Round 12 integrated frontend

One KV group receives 32-byte ordered summary bursts and produces four signed 48-bit Q24 scores together. Sixteen channel lanes feed each Q head. The four heads share each code chunk, while each head keeps its own query, dot accumulators and scale products. The RTL exercises 8, 6, 5 and 4 chunks, corresponding to 128, 96, 80 and 64 channels.

The stream carries the channel codes, two little-endian FP16 scales, a uint8 length and alignment padding. A 112-byte or 80-byte stride shares alternating bursts across adjacent records. The 96-byte stride is aligned to three bursts. A shared beat writes a 16-byte half to each record bank, with no repeated burst. A last-beat marker closes the epoch and discards the unused half when the final record ends mid-burst. Length zero marks a masked score. The bias ROM covers valid page lengths 1–32.

The two record banks hold a record from its first incoming byte through its last dot chunk. The first chunk requires a full record and all eight coefficients. The coefficient lane accepts one operand each cycle and its result is registered through nine stages. One lane issues eight requests in eight cycles. Two lanes issue eight in four cycles. The measured interval from the first coefficient issue to the last result write is 16 cycles with one lane and 12 cycles with two. The result FIFO reserves a credit before the first dot chunk and releases it on the output handshake. Each of its two entries carries 192 score bits, a 32-bit record tag and an 8-bit count. A query epoch can start once the previous epoch’s final output is consumed.

## RTL service

The steady interval is between accepted complete records, under continuous input and an accepting output. The first-result latency starts at the first accepted burst or preloaded transaction. The RTL’s output is a single group handshake, so the first and last head latencies coincide.

| Profile | Stride | Dot chunks | Coefficients per cycle | Record interval pattern | Effective group II | First result, cycles | Last dot to output, cycles |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Preloaded 96 | 112 B | 6 | injected | 6, 7 | 6.5 | 14 | 8 |
| Full 128 | 144 B | 8 | 2 | 4, 26 | 15 | 33 | 8 |
| Full 96 | 112 B | 6 | 1 | 3, 29 | 16 | 34 | 8 |
| Full 96 | 112 B | 6 | 2 | 3, 23 | 13 | 30 | 8 |
| Patched 96 | 112 B | 6 | 2 | 3, 23 | 13 | 32 | 10 |
| Full 80 | 96 B | 5 | 1 | 8, 17 | 12.5 | 32 | 8 |
| Full 80 | 96 B | 5 | 2 | 5, 16 | 10.5 | 28 | 8 |
| Full 64 | 80 B | 4 | 1 | 2, 28 | 15 | 31 | 8 |
| Full 64 | 80 B | 4 | 2 | 2, 20 | 11 | 27 | 8 |

The preloaded case receives the whole record and eight coefficients in one 984-bit transaction. Its stride is listed for the corresponding memory layout. The 96-channel dual issue path needs 27 cycles from the first complete record to the first output. The patched path needs 29. Both release their first input slot 19 cycles after that record completes. The aligned 80-channel layout avoids reserving the next bank to finish a shared beat, which changes the observed cadence.

The scoreboard checks exact coefficients before the first dot, all four scores and count on output, record ordering, masked length, query epoch immutability, shared beats, and held outputs under backpressure. Every integrated profile checked 524 head scores per synthetic run and 408 more across six local calibration epochs. Synthetic pressure tests reach two occupied record banks and two occupied result entries. The legacy single-head test checked 524 synthetic and 408 calibration head scores with its original streamed-chunk interface. Its first accepted chunk to visible score takes 9 cycles with an II of 8 cycles per head.

[RTL validation receipt](rtl_validation.json)

The logical input capacity at 96 channels is two 112-byte record banks, including the header and padding. Each bank has an aggregate beat write up to 32 bytes and a selected 16-byte code-chunk read. The scale and count fields have direct outputs for preparation, while each bank keeps eight 22-bit prepared coefficients. Query storage has one 512-bit wide raw bank and one magnitude bank per chunk, plus four binary64 gains. The result FIFO stores 58 bytes and has one read and one write port. The frontend assumes an ordered burst stream from upstream and a single clock. The evidence does not include an addressed response reordering queue, a CDC bridge or SRAM macro timing.

The depth sensitivity matched the two-slot RTL schedule, then replayed the same bytes with four record slots. With two result credits, the sampled intervals averaged 6.57 cycles in the original tail and 7.53 with the patched tail. With both four record slots and three result credits, both averaged 6.56 cycles. These are cycle-model estimates with unchanged arithmetic and larger storage.

[Record-depth sensitivity receipt](depth_sensitivity.json)

## Standard-cell mapping

All mapped cases use NangateOpenCellLibrary typical, a 1.25 ns clock constraint, BUF_X1 input drive and 5 fF output load. Rates use the reported minimum setup period and measured continuous cadence. They are pre-layout estimates.

| Case | Mapped area, µm² | Critical period, ns | WNS at 1.25 ns, ns | Head scores per second, millions | Four-head records per second, millions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy 128, one head | 35,524.034 | 2.230466 | −0.980466 | 56.04 | 14.01 with serial head reuse |
| Preloaded 96, four heads | 163,664.480 | 1.903731 | −0.653731 | 323.25 | 80.81 |
| Full 96, four heads, dual coefficient | 185,390.828 | 1.881502 | −0.631502 | 163.54 | 40.88 |
| Patched 96, four heads, dual coefficient | 192,827.656 | 1.741593 | −0.491593 | 176.67 | 44.17 |

Four independent legacy replicas would occupy 142,096.136 µm² and process 56.04 million four-head records per second under ideal parallel feeding. That estimate uses the legacy scorer’s precomputed coefficients and its chunk-stream ingress. The full 96-channel patch costs 4.01% mapped area over the original and improves the critical period by 7.44%, while leaving the two-slot record cadence at 13 cycles. All four cases miss 1.25 ns.

The original integrated critical path ran from a saved radius dot sum into a full-width coefficient multiply. The score and bias addition was the next slow endpoint group. The patch splits each coefficient into two 11-bit partial products and registers a carry-save compression of the scaled score terms and bias. The new critical path still lands in the scaling stage, from saved coefficient metadata to an 11-bit radius partial multiply. The by-stage endpoints, slack and cell area appear in the synthesis receipt. Cell area is apportioned by mapped register endpoint and its input logic cone, with shared cones accounted separately.

[Patch basis](patch_basis.json) · [Synthesis receipt](synthesis.json) · [Legacy STA](sta/old128_sta.txt) · [Preloaded STA](sta/preloaded96_sta.txt) · [Full STA](sta/full96_dual_sta.txt) · [Patched STA](sta/patched96_dual_sta.txt)

## Selected 80-channel profile

Calibration selected N80. Its dual-issue split-tail version held the 5, 16 complete-record cadence for an effective 10.5-cycle II. It accepted the first group result 30 cycles after the first ingress beat, 28 cycles after the first complete record and 10 after the last dot. The scoreboard checked 524 continuous synthetic, 524 stalled synthetic and 408 captured head scores.

The mapped area was 181,390.986 µm² and the critical period was 1.746854 ns. The 1.25 ns setup slack was −0.496854 ns. The worst path starts at the registered carry-save term and ends at the final result register. At its STA period, the measured cadence projects 54.52 million four-head records per second. That is 23.4% above the patched 96-channel rate with 5.93% less mapped area.

[N80 RTL validation](narrow_rtl_validation.json) · [N80 synthesis](narrow_synthesis.json) · [N80 STA](sta/patched80_dual_sta.txt)

## Reproduce

```bash
.venv/bin/python scripts/check_round12_rtl.py
.venv/bin/python scripts/check_round12_depth.py
.venv/bin/python scripts/synth_round12.py --jobs 4 \
  --rtl-validation results/round12_frontend/hardware/rtl_validation.json
.venv/bin/python scripts/check_round12_rtl.py --profile patched80_dual \
  --result results/round12_frontend/hardware/narrow_rtl_validation.json
.venv/bin/python scripts/synth_round12.py --case patched80_dual \
  --rtl-validation results/round12_frontend/hardware/narrow_rtl_validation.json \
  --result results/round12_frontend/hardware/narrow_synthesis.json
```

Add `--fixture-dir results/round12_frontend/local_inputs` to the RTL command for the local calibration vectors. The synthetic vectors are generated deterministically by the script. The RTL receipt records the local vector hashes and the Icarus run logs are under `.cache/round12/rtl`. The synthesis scripts, netlists, and tool logs are under `.cache/round12/synth`.
