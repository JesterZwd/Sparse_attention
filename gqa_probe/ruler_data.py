"""Pinned legacy RULER v1 input adapter; never truncates an official prompt.

The official answer prefix is appended to the assistant generation header, as in
RULER's model templates. All evidence offsets refer to the resulting token IDs.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

ANSWER_PREFIX_START = ' The special magic'


def render_chat(tokenizer, user_input: str, answer_prefix: str) -> str:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_input}], tokenize=False,
        add_generation_prompt=True,
    ) + answer_prefix


class ChatBudgetTokenizer:
    """RULER tokenizer protocol, counting the final chat and answer prefix."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def text_to_tokens(self, text: str) -> list[int]:
        prefix_at = text.rfind(ANSWER_PREFIX_START)
        if prefix_at < 0:
            # Upstream also calls this for its obsolete first-answer position.
            user_input, answer_prefix = text, ''
        else:
            user_input, answer_prefix = text[:prefix_at], text[prefix_at:]
        return self.tokenizer.encode(
            render_chat(self.tokenizer, user_input, answer_prefix),
            add_special_tokens=False,
        )


def select_chat_tokenizer(tokenizer_type: str, tokenizer_path: str) -> ChatBudgetTokenizer:
    """Import hook used by the preserved, one-line-patched official generator."""
    if tokenizer_type != 'hf':
        raise ValueError('The pinned Round 2 adapter requires the HF tokenizer')
    from transformers import AutoTokenizer
    return ChatBudgetTokenizer(AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True,
    ))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _token_span(offsets, start: int, end: int) -> list[int]:
    indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not indices or offsets[indices[0]][0] > start or offsets[indices[-1]][1] < end:
        raise ValueError('source association is not fully represented by final tokens')
    return [indices[0], indices[-1] + 1]


def finalize_row(raw: dict, tokenizer, *, task: str, split: str, seed: int,
                 max_seq_length: int = 16384, max_new_tokens: int = 128,
                 source_metadata: dict | None = None) -> dict:
    """Map every queried association/value from an unmodified official row.

    ``gold_spans`` covers complete source key–value sentences, including their
    values. Nested ``gold_evidence`` also records the narrower value spans and
    character intervals for separate value-only recall and independent audits.
    """
    references = list(raw['outputs'])
    if not references or any(not isinstance(value, str) or not value for value in references):
        raise ValueError('nonempty official references are required')
    prefix = raw['answer_prefix']
    query_match = re.fullmatch(
        r' The special magic (?:number|numbers|uuid|uuids|word|words) for (.+?) '
        r'mentioned in the provided text (?:is|are)', prefix,
    )
    if query_match is None:
        raise ValueError('unsupported official answer prefix')
    key = query_match.group(1)
    text = render_chat(tokenizer, raw['input'], prefix)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if len(ids) + max_new_tokens > max_seq_length:
        raise ValueError(f'final chat/output budget exceeded: {len(ids)}+{max_new_tokens}>{max_seq_length}')
    user_start = text.find(raw['input'])
    if user_start < 0:
        raise ValueError('chat template did not preserve the official source input')
    evidence = []
    for value in references:
        pattern = re.compile(
            r'One of the special magic (?:numbers|uuids|words) for '
            + re.escape(key) + r' is: (?P<value>' + re.escape(value) + r')\.'
        )
        matches = list(pattern.finditer(raw['input']))
        if not matches:
            raise ValueError(f'missing source association for official reference {value!r}')
        for match in matches:
            association_chars = [user_start + match.start(), user_start + match.end()]
            value_chars = [user_start + match.start('value'), user_start + match.end('value')]
            evidence.append({
                'key': key, 'value': value,
                'association_span': _token_span(offsets, *association_chars),
                'value_span': _token_span(offsets, *value_chars),
                'association_chars': association_chars, 'value_chars': value_chars,
            })
    metadata = dict(source_metadata or {})
    metadata.update({
        'seed': seed, 'official_index': raw['index'], 'query_key': key,
        'instance_sha256': _sha256_json({'key': key, 'values': sorted(references)}),
        'final_chat_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'input_ids_sha256': _sha256_json(ids),
        'gold_evidence': evidence,
        'gold_span_semantics': 'complete queried source key-value associations; value spans nested in gold_evidence',
        'official_token_position_answer_ignored': True,
    })
    return {
        'sample_id': f'{split}/{task}/{raw["index"]:04d}',
        'task': task, 'split': split, 'input_ids': ids,
        'references': references,
        'gold_spans': [item['association_span'] for item in evidence],
        'input_tokens': len(ids), 'max_new_tokens': max_new_tokens,
        'source_metadata': metadata,
    }


def validate_splits(rows: Iterable[dict], max_seq_length: int = 16384) -> None:
    """Reject duplicated source instances/prompts/IDs and cross-split seeds."""
    instances, prompts, ids, seeds = set(), set(), set(), {}
    for row in rows:
        metadata = row['source_metadata']
        instance = metadata['instance_sha256']
        if instance in instances:
            raise ValueError('duplicate retrieval instance')
        instances.add(instance)
        if row['sample_id'] in ids:
            raise ValueError('duplicate sample ID')
        ids.add(row['sample_id'])
        prompt_hash = _sha256_json(row['input_ids'])
        if prompt_hash in prompts:
            raise ValueError('duplicate final input')
        prompts.add(prompt_hash)
        seed = metadata['seed']
        previous_split = seeds.setdefault(seed, row['split'])
        if previous_split != row['split']:
            raise ValueError('random seed reused across splits')
        n = len(row['input_ids'])
        if n != row['input_tokens'] or n + row['max_new_tokens'] > max_seq_length:
            raise ValueError('final chat/output budget exceeded')
        if not row['gold_spans'] or any(not (0 <= a < b <= n) for a, b in row['gold_spans']):
            raise ValueError('invalid final gold span')
