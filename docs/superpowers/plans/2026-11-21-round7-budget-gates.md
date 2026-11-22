# Round 7 implementation plan

1. Add a single K reservation and protected-prefix V fallback. Verify physical burst limits on both sides of a page boundary, final mask and cumulative loss denominators. Preserve the Round 6 dynamic behavior and add its 68-page cap control.
2. Add exact, 64-bin, running-peak and checked-threshold V gates. Compare the four on identical candidate logits, then send only qualified points to the paired closure.
3. Reuse the existing Round 6 sixteen-prompt calibration and untouched 128-prompt seed. Pair Oracle A1, INT4 and Mean6+Radius2 at the same full-V K count. Run both 144 B summaries with identical hard cap and exact Top-p at 5%. Record the chosen cap separately from observed bytes.
4. Sample common states for reuse and cold mass diagnostics. Keep the official key/value evidence separate from output error and label dependency timing and selector storage as modeled.
5. Lock the active policy and strongest compliant sparse control before accessing test predictions. Use zero violations, paired 95% noninferiority bound of −1 percentage point and 5% modeled dependency cycle or control SRAM reduction. Gate integer datapath, streaming, RTL and model scaling on evidence.
6. Run regression, validate public artifacts, and fast-forward GitHub main without raw inputs, Q/K/V traces, weights or machine-specific logs.
