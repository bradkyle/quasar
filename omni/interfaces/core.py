# exposes the core functionality to each interface for use

import json
import requests_cache
import requests
import time
from omni.interfaces.invoke import invoke

prev_balances = {}
bad_request_penalty = 0
bad_connection_penalty = 0
none_response_penalty = 0
not_afforded_penalty = 0
rate_limit_penalty = 0
step_penalty = 0
not_found_penalty = 0
affordance_disabled_penalty = 0
response_size_penalty = 0
current_gemini_order_id = 0
wait_penalty = 0


session = requests_cache.CachedSession(cache_name="cryptonator", expire_after=30, backend='sqlite')

#todo this should account for the fact that prices might not have enough time to change between steps with respect to cache
def profit_over_time(id, balance_value):
    if id in prev_balances:
        prev_balance = prev_balances[id]
        profit = float(balance_value) - float(prev_balance["value"])
        profit_time = profit / (time.time() - prev_balance["time"])
    else:
        profit_time = 0.0
    prev_balances[id] = {"value": balance_value, "time": time.time()}
    return profit_time

def profit(id, balance_value):
    if id in prev_balances:
        prev_balance = prev_balances[id]
        profit = float(balance_value) - float(prev_balance["value"])
    else:
        profit = 0.0
    prev_balances[id] = {"value": balance_value, "time": time.time()}
    return profit

def calc_value(symbol, balance): #todo create asynchronous request to multiple services
    response = json.loads(invoke("GET", url="https://api.cryptonator.com/api/ticker/{}-usd".format(symbol), encode=False, session=session))
    value = float(response["ticker"]["price"]) * float(balance)
    return value

def projected_return_on_current_balance():
    raise NotImplemented

def projected_return_on_amount(): # $100, $1000, $10000 * 1 day, 1 week, 1 month, 1 year
    raise NotImplemented

def average_profit_per_step_aggregated():
    raise NotImplemented

def average_profit_per_step_seperate():
    raise NotImplemented

def average_profit_over_time_aggregated():
    raise NotImplemented

def average_profit_over_time_seperate():
    raise NotImplemented

def get_nonce():
    return time.time() * 1000


