"""Run the Round 11 Qwen3 32K Multi-Value calibration only."""

from pathlib import Path

import run_calibration as runner


if __name__ == '__main__':
    data = Path('.cache/comrad-round14/data/qwen3_32k_mv')
    expected = '95407597972e422f0da09a35ab0dfd8063cc53ecc364e2915111b73cc2dbc2c7'
    if runner._sha(data / 'calibration.jsonl') != expected:
        raise ValueError('32K calibration rows differ from the Round 11 receipt')
    runner.PANELS['qwen3_32k_mv'] = data
    runner.main()
