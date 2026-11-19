"""Dataset boundary tests: no model loads or generated-answer inspection."""
import copy
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained('models/Qwen3-4B-Instruct-2507', local_files_only=True)


def raw_four():
    return {
        'index': 0,
        'input': 'Memorize.\nOne of the special magic numbers for amber-apple is: 1234567.\n'
                 'One of the special magic numbers for amber-apple is: 2345678.\n'
                 'One of the special magic numbers for amber-apple is: 3456789.\n'
                 'One of the special magic numbers for amber-apple is: 4567890.\n'
                 'What are all the special magic numbers for amber-apple mentioned in the provided text?',
        'answer_prefix': ' The special magic numbers for amber-apple mentioned in the provided text are',
        'outputs': ['1234567', '2345678', '3456789', '4567890'],
    }


def make_row(tokenizer, raw=None, **kwargs):
    from gqa_probe.ruler_data import finalize_row
    return finalize_row(raw or raw_four(), tokenizer, task='niah_multivalue',
                        split=kwargs.pop('split', 'engineering'), seed=kwargs.pop('seed', 11), **kwargs)


def test_final_chat_keeps_official_answer_prefix_in_assistant_continuation(tokenizer):
    row = make_row(tokenizer)
    text = tokenizer.decode(row['input_ids'])
    assert text == '<|im_start|>user\n' + raw_four()['input'] + '<|im_end|>\n<|im_start|>assistant\n' + raw_four()['answer_prefix']
    assert row['input_tokens'] == len(row['input_ids'])
    assert row['max_new_tokens'] == 128


def test_all_four_values_and_associations_are_mapped_after_final_chat_tokenization(tokenizer):
    row = make_row(tokenizer)
    evidence = row['source_metadata']['gold_evidence']
    assert [item['value'] for item in evidence] == ['1234567', '2345678', '3456789', '4567890']
    for item in evidence:
        assert tokenizer.decode(row['input_ids'][slice(*item['value_span'])]) == item['value']
        assert tokenizer.decode(row['input_ids'][slice(*item['association_span'])]).strip() == (
            'One of the special magic numbers for amber-apple is: ' + item['value'] + '.')
        assert item['association_span'][0] > 3
    assert row['gold_spans'] == [item['association_span'] for item in evidence]


def test_gold_match_uses_queried_key_not_an_earlier_distractor_value(tokenizer):
    raw = raw_four()
    raw['input'] = 'One of the special magic numbers for blue-banana is: 1234567.\n' + raw['input']
    row = make_row(tokenizer, raw)
    first = row['source_metadata']['gold_evidence'][0]
    decoded = tokenizer.decode(row['input_ids'][slice(*first['association_span'])])
    assert 'amber-apple' in decoded
    assert 'blue-banana' not in decoded


def test_budget_includes_128_output_tokens_and_refuses_to_truncate_needles(tokenizer):
    row = make_row(tokenizer)
    exact = row['input_tokens'] + 128
    assert make_row(tokenizer, max_seq_length=exact)['input_ids'] == row['input_ids']
    with pytest.raises(ValueError, match='budget'):
        make_row(tokenizer, max_seq_length=exact - 1)


def test_answer_only_in_question_cannot_fabricate_a_source_span(tokenizer):
    raw = raw_four()
    raw['input'] = raw['input'].replace('is: 1234567.', 'is: 9999999.') + ' Answer 1234567?'
    with pytest.raises(ValueError, match='source association'):
        make_row(tokenizer, raw)


def test_template_budget_tokenizer_counts_exact_final_chat_with_prefix(tokenizer):
    from gqa_probe.ruler_data import ChatBudgetTokenizer
    raw = raw_four()
    budget_tokenizer = ChatBudgetTokenizer(tokenizer)
    assert budget_tokenizer.text_to_tokens(raw['input'] + raw['answer_prefix']) == make_row(tokenizer)['input_ids']


def test_splits_reject_duplicate_instances_even_when_prompt_or_seed_differs(tokenizer):
    from gqa_probe.ruler_data import validate_splits
    engineering = make_row(tokenizer)
    raw = raw_four()
    raw['input'] = 'Different background. ' + raw['input']
    calibration = make_row(tokenizer, raw, split='calibration', seed=22)
    with pytest.raises(ValueError, match='instance'):
        validate_splits([engineering, calibration])


def test_splits_reject_reused_seeds_for_different_retrieval_instances(tokenizer):
    from gqa_probe.ruler_data import validate_splits
    engineering = make_row(tokenizer)
    raw = raw_four()
    raw['input'] = raw['input'].replace('amber-apple', 'copper-carrot')
    raw['answer_prefix'] = raw['answer_prefix'].replace('amber-apple', 'copper-carrot')
    calibration = make_row(tokenizer, raw, split='calibration', seed=11)
    with pytest.raises(ValueError, match='seed'):
        validate_splits([engineering, calibration])


def test_disjoint_splits_pass_and_ids_include_split_and_task(tokenizer):
    from gqa_probe.ruler_data import validate_splits
    a = make_row(tokenizer)
    raw = raw_four()
    raw['input'] = raw['input'].replace('amber-apple', 'copper-carrot')
    raw['answer_prefix'] = raw['answer_prefix'].replace('amber-apple', 'copper-carrot')
    b = make_row(tokenizer, raw, split='test', seed=22)
    validate_splits([a, b])
    assert a['sample_id'] == 'engineering/niah_multivalue/0000'
    assert b['sample_id'] == 'test/niah_multivalue/0000'


def test_scoring_preserves_pinned_official_fractional_metric_and_separate_complete_flag():
    from gqa_probe.paired_statistics import score_prediction
    official_path = Path('.cache/ruler-upstream/scripts/eval/synthetic/constants.py')
    if not official_path.exists():
        pytest.skip('optional upstream parity check requires the pinned RULER checkout')
    official = {}
    exec(compile(official_path.read_text(), str(official_path), 'exec'), official)
    predictions = ['ABC 2345678', 'nothing', '1234567 2345678 3456789 4567890']
    references = [['abc', '2345678', 'third', 'fourth'], ['1234567'], raw_four()['outputs']]
    scores = [score_prediction(prediction, refs) for prediction, refs in zip(predictions, references)]
    assert [score['official_score'] for score in scores] == [.5, 0., 1.]
    assert [score['complete_correct'] for score in scores] == [False, False, True]
    assert official['TASKS']['niah']['metric_fn'](predictions, references) == 50.
    assert round(sum(score['official_score'] for score in scores) / 3 * 100, 2) == 50.


def test_uuid_value_span_covers_shared_leading_space_bpe_token(tokenizer):
    from gqa_probe.ruler_data import finalize_row
    value = 'c1d55926-ca53-4511-ba72-26474588f75b'
    key = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
    raw = {'index': 0,
           'input': 'One of the special magic uuids for ' + key + ' is: ' + value + '.\nWhat is the uuid?',
           'outputs': [value],
           'answer_prefix': ' The special magic uuid for ' + key + ' mentioned in the provided text is'}
    row = finalize_row(raw, tokenizer, task='niah_multikey_3', split='engineering', seed=33)
    evidence = row['source_metadata']['gold_evidence'][0]
    # Fast-tokenizer offsets cover entire BPE tokens; a leading space belongs
    # to the same token as the UUID's initial letter and must not be discarded.
    decoded = tokenizer.decode(row['input_ids'][slice(*evidence['value_span'])])
    assert decoded == ' ' + value
    final = tokenizer.decode(row['input_ids'])
    assert final[slice(*evidence['value_chars'])] == value
