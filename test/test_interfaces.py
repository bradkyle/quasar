import unittest
from omni.interfaces.registration import closer_registry

class TestInterfaces(unittest.TestCase):
    def setUp(self):
        return NotImplemented

    def test_blockchain_get_chart(self):
        return NotImplemented

    def test_blockchain_get_ticker(self):
        return NotImplemented

    def test_blockchain_get_stats(self):
        return NotImplemented

    def test_blockchain_get_pools(self):
        return NotImplemented

    def test_get_symbols(self):
        return NotImplemented

    def test_get_ticker(self):
        return NotImplemented

    def test_get_orderbook(self):
        return NotImplemented

    def test_get_current_auction(self):
        return NotImplemented

    def test_auction_history(self):
        return NotImplemented

    def test_profit_over_time(self):
        return NotImplemented

    def tearDown(self):
        closer_registry.close()

