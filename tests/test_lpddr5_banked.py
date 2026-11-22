import pytest

from gqa_probe.lpddr5_banked import read_banked


def test_open_row_reads_use_same_bank_group_spacing():
    replay = read_banked([[(0, 64)]])

    # ACT1=2, ACT2=3, RD=18/22, data return=37/41.
    assert replay["phase_first_command_ck"] == [2]
    assert replay["phase_first_read_ck"] == [18]
    assert replay["phase_end_ck"] == [41]
    assert (replay["bursts"], replay["row_hits"], replay["activations"], replay["precharges"]) == (2, 1, 1, 0)
    assert replay["command_cycles"] == 4


def test_two_banks_can_activate_while_the_older_read_waits_for_trcd():
    phases = [[(0, 32), (2048, 32)]]
    parallel = read_banked(phases, descriptor=0)
    serial = read_banked(phases, descriptor=0, queue_depth=1)

    assert parallel["phase_end_ck"] == [39]  # ACT2=1/5, RD=16/20, data=35/39.
    assert serial["phase_end_ck"] == [70]
    assert parallel["max_outstanding"] == 2
    assert serial["max_outstanding"] == 1
    assert parallel["activations"] == serial["activations"] == 2


def test_row_conflicts_honor_ras_rp_and_rc_instead_of_reordering_within_bank():
    replay = read_banked([[(0, 32), (32768, 32)]], descriptor=0)

    # First ACT2=1, RD=16, PRE=35, second ACT2=50, RD=65, data=84.
    assert replay["phase_end_ck"] == [84]
    assert (replay["row_hits"], replay["activations"], replay["precharges"]) == (0, 2, 1)
    assert replay["command_cycles"] == 7


def test_dependency_waits_use_last_data_return_and_preserve_open_row():
    replay = read_banked([[(0, 32)], [], [(32, 32)]], wait=[5, 9, 4])

    assert replay["phase_end_ck"] == [42, 51, 76]
    assert replay["phase_first_command_ck"] == [7, None, 57]
    assert replay["phase_first_read_ck"] == [23, None, 57]
    assert (replay["bursts"], replay["row_hits"], replay["activations"]) == (2, 1, 1)


def test_credit_window_includes_reads_that_have_not_returned():
    replay = read_banked([[(0, 4096)]], descriptor=0, queue_depth=8)

    assert replay["max_outstanding"] == 8
    assert replay["bursts"] == 128
    assert replay["row_hits"] + replay["activations"] == replay["bursts"]


def test_request_ranges_must_be_burst_aligned():
    with pytest.raises(ValueError, match="32-byte"):
        read_banked([[(0, 31)]])


def test_rank_wide_precharge_spacing():
    # Conflicts for four different open banks all become eligible after
    # the first phase drains. Consecutive PRE commands must honor nPPD.
    first = [(bank * 2048, 32) for bank in range(4)]
    second = [(32768 + bank * 2048, 32) for bank in range(4)]
    result = read_banked([first, second], descriptor=0)
    assert result["precharges"] == 4
    assert result["minimum_precharge_gap_ck"] == 2
