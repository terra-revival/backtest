from pytest import approx
from terra_algo_backtest.brown import simulationSamples

from .test_exec_engine import assert_exec_info


class TestBrown:
    def setup_method(self):
        pass

    def test_simulationSamples(self):
        functions = simulationSamples(1420, False, 365, 24 * 60 * 60)
        assert type(functions) == dict
        for sim in functions:
            assert 'data' in functions[sim]
            assert 'timeframe' in functions[sim]
            assert functions[sim]['timeframe'] == 24 * 60 * 60
            assert len(functions[sim]['data']) == 365 + 1
