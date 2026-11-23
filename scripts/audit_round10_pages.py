"""Export chosen 32K K pages and retained V pages from frozen common states."""
import argparse
from pathlib import Path
from unittest.mock import patch

import torch

from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts import run_round10 as experiment
from scripts.round7_run import DENSE
from gqa_probe.round7_attention import Attention


@torch.inference_mode()
def export(root):
    manifest, rows = experiment.dataset(root, "32k", "calibration")
    model, tokenizer, eos, model_sha = experiment.init_model(manifest)
    masks = experiment.read_masks(root, "cuda")
    diagnostics = root / "diagnostic32"
    source = diagnostics / "bundles"
    destination = diagnostics / "selected_pages"
    destination.mkdir(exist_ok=True)
    receipts = {}
    evaluate = experiment.candidate
    for index, row in enumerate(rows[experiment.FIT:], experiment.FIT):
        filename = f"{index:04d}.json.gz"
        old = experiment.read_bundle(source / filename)
        if old["sample_id"] != row["sample_id"]:
            raise ValueError("common-state identity changed")
        output = destination / filename
        if not output.exists():
            cache = experiment.prefill(model, row)
            args = model, cache, row["max_new_tokens"], eos
            dense = generate_from_prefill(*args, Attention(DENSE, None, row["gold_spans"]))
            starts = experiment.answer_spans(tokenizer, dense.token_ids, row)
            if sorted(starts) != old["first_value_decode_steps"]:
                raise ValueError("answer token timing changed")
            probe = experiment.DiagnosticAttention(masks, experiment.original_spans(row), starts)
            recorded = []

            def capture(*args, **kwargs):
                allocation, result = evaluate(*args, **kwargs)
                recorded.append((allocation.ids, result.union.any(-1)))
                return allocation, result

            with patch.object(experiment, "candidate", capture):
                replay = generate_from_prefill(*args, probe)
            if replay.token_ids != dense.token_ids or len(probe.rows) != len(old["rows"]) or len(recorded) != len(probe.rows):
                raise ValueError("common-state generation or evaluation count changed")
            decisions = []
            for prior, actual, (pages, keep_v) in zip(old["rows"], probe.rows, recorded, strict=True):
                identity = {k: actual[k] for k in ("phase", "policy", "step", "layer", "page", "answer_start")}
                if any(prior[k] != v for k, v in identity.items()):
                    raise ValueError("probe identity changed")
                for ledger in ("total", "k_records", "v_records", "k_page_hits", "v_gold_token_hits"):
                    if prior[ledger] != actual[ledger].tolist():
                        raise ValueError(f"common-state {ledger} changed")
                groups = pages.tolist()
                retained = keep_v.tolist()
                decisions.append(dict(**identity, k_pages=groups, prefix_pages=actual["prefix_pages"].tolist(),
                                      v_pages=[[p for p, keep in zip(ids, selected, strict=True) if keep]
                                               for ids, selected in zip(groups, retained, strict=True)]))
            experiment.prompt_bundle(output, dict(sample_id=row["sample_id"], selections=decisions))
            del cache, probe, recorded
        receipts[filename] = dict(bundle_sha256=_sha(output), reference_sha256=_sha(source / filename))
    _write_json(destination / "complete.json", dict(
        source_sha256={**experiment.sources(), __file__: _sha(Path(__file__))},
        dataset_sha256=_sha(root / "data" / "32k" / "manifest.json"),
        model_sha256=model_sha, channels_sha256=_sha(root / "channels.json"), bundles=receipts))
    print(f"Verified selected K and retained V page IDs for {len(receipts)} common-state prompts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    args = parser.parse_args()
    export(args.root)


if __name__ == "__main__":
    main()
