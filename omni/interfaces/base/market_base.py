import json
import requests_cache
import requests
import time
from omni.interfaces.invoke import invoke


# todo implement store instead as well as alternative sources of crypto information

class MarketBase(object):
    def __init__(self):
        self.prev_balances = {}
        self.session = requests_cache.CachedSession(cache_name="cryptonator", expire_after=30, backend='sqlite')

    def calc_value(self, symbol, balance):
        response = json.loads(invoke("GET", url="https://api.cryptonator.com/api/ticker/{}-usd".format(symbol), encode=False, session=self.session))
        value = float(response["ticker"]["price"]) * float(balance)
        return value

    #todo this should account for the fact that prices might not have enough time to change between steps

    def profit_over_time(self, id, balance_value):
        if id in self.prev_balances:
            prev_balance = self.prev_balances[id]
            profit = float(balance_value) - float(prev_balance["value"])
            profit_time = profit / (time.time() - prev_balance["time"])
        else:
            profit_time = 0.0
        self.prev_balances[id] = {"value": balance_value, "time": time.time()}
        return profit_time

    def profit(self, id, balance_value):
        if id in self.prev_balances:
            prev_balance = self.prev_balances[id]
            profit = float(balance_value) - float(prev_balance["value"])
        else:
            profit = 0.0
        self.prev_balances[id] = {"value": balance_value, "time": time.time()}
        return profit


market_base = MarketBase()