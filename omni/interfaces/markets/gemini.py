from omni.interfaces.invoke import invoke
from omni.interfaces.processing import process
from omni.interfaces.core import get_nonce, switch
from omni.interfaces.config import MAX_ORDER_SIZE, MAX_ORDER_PRICE
import base64
import hashlib
import hmac
import json
import time
from omni.interfaces.base.market_base import market_base
from omni.interfaces.registration import affordance
import requests
import requests_cache

API_VERSION = '/v1'
BASE_URI = "https://api.sandbox.gemini.com" + API_VERSION

#todo has to account for multiple accounts
def _get_next_order_id():
    from omni.interfaces.core import current_gemini_order_id
    current_gemini_order_id += 1
    return current_gemini_order_id

def _invoke_api(endpoint, payload, params=None, pub=True, keys=None, encode=True):

    public_session = requests_cache.CachedSession(cache_name="gemini_public", expire_after=30, backend='sqlite')
    private_session =requests_cache.CachedSession(cache_name="gemini_private", expire_after=30,backend='sqlite')

    url = BASE_URI + endpoint

    load = payload

    if pub == False:
        if keys is None:
            raise Exception

        # base64 encode the payload
        payload = str.encode(json.dumps(payload))
        b64 = base64.b64encode(payload)

        # sign the requests
        signature = hmac.new(str.encode(keys['private']), b64, hashlib.sha384).hexdigest()

        headers = {
            'Content-Type': 'text/plain',
            'X-GEMINI-APIKEY': keys['public'],
            'X-GEMINI-PAYLOAD': b64,
            'X-GEMINI-SIGNATURE': signature
        }

        return invoke("POST", url=url, headers=headers, payload=load, session=private_session, encode=encode)
    else:
        return invoke("GET", url=url, params=params, payload=load, session=public_session, encode=encode)



# Public API methods
# ------------------
# State
def get_symbols(input):
    """ https://docs.gemini.com/rest-api/#symbols """
    endpoint = '/symbols'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, pub=True)


def get_ticker(input):
    """ https://docs.gemini.com/rest-api/#ticker """
    endpoint = '/pubticker/' + input.symbol

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, pub=True)


def get_order_book(input):
    """ https://docs.gemini.com/rest-api/#current-order-book """
    endpoint = '/book/' + input.symbol

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, pub=True)


def get_trade_history(input):


    params = {}
    # params['since'] = since
    # params['limit_trades'] = limit_trades
    # params['include_breaks'] = include_breaks

    endpoint = '/trades/' + input.symbol

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, params, pub=True)



def get_current_auction(input):
    """ https://docs.gemini.com/rest-api/#current-aucion """
    endpoint = '/auction/' + input.symbol

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, pub=True)



def get_auction_history(input):

    params = {}
    # params['since'] = since
    # params['limit_auction_results'] = limit_auction_results
    # params['include_indicative'] = include_indicative

    endpoint = '/auction/' + input.symbol + '/history'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, params, pub=True)



# Order Status API
# https://docs.gemini.com/rest-api/#order-status
# ----------------------------------------------
# State
def get_active_orders(input):

    endpoint = '/orders'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)



def get_order_status(input):

    """ https://docs.gemini.com/rest-api/#order-status """
    endpoint = '/order/status'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce(),
        'order_id': input.order_id
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)



def get_trade_volume(input):

    endpoint = '/tradevolume'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)



def get_past_trades(input):

    endpoint = '/mytrades'

    payload = {
        'request': API_VERSION + endpoint,
        'symbol': input.symbol,
        'nonce': get_nonce(),
        'limit_trades': input.limit_trades,
        'timestamp': input.timestamp
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)



# Order Placement API
# https://docs.gemini.com/rest-api/#new-order
# -------------------------------------------

def new_order(input):

    client_order_id = str(_get_next_order_id())
    price = switch(input.price, round((input.args[0] * float(MAX_ORDER_PRICE)), 2))
    amount = switch(input.amount, round((input.args[1] * MAX_ORDER_SIZE), 2))
    endpoint = '/order/new'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce(),
        'client_order_id': client_order_id,
        'symbol': input.symbol,
        'amount': amount,
        'price': price,
        'side': input.side,
        'type': 'exchange limit'
    }

    if input.options != "" or None:
        payload['options'] = [input.options]

    response = _invoke_api(endpoint, payload, keys=input.key_set, pub=False, encode=False)
    if response is not None:
        loaded_response = json.loads(response)

        print("response:" + str(response))

        if "result" not in loaded_response:
            new_order_id = loaded_response[0]["order_id"]
            affordance(entry_point='omni.interfaces.markets.gemini:get_order_status', key_set=input.key_set, order_id=new_order_id)
            affordance(entry_point='omni.interfaces.markets.gemini:cancel_order', key_set=input.key_set, order_id=new_order_id)
        # todo else

        return process(response)
    #todo else:


def cancel_order(input):

    endpoint = '/order/cancel'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce(),
        'order_id': input.order_id
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)

def cancel_session_orders(input):
    """ https://docs.gemini.com/rest-api/#cancel-all-session-orders """
    endpoint = '/order/cancel/session'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)

def cancel_all_orders(input):
    """ https://docs.gemini.com/rest-api/#cancel-all-active-orders """
    endpoint = '/order/cancel/all'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)

# Fund Management API's
# https://docs.gemini.com/rest-api/#get-available-balances
# --------------------------------------------------------
def get_balance(input):
    """ https://docs.gemini.com/rest-api/#get-available-balances """
    endpoint = '/balances'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    return _invoke_api(endpoint, payload, keys=input.key_set, pub=False)

# Tasks
# =====================================================================================================================>

def profit_over_time(input):
    endpoint = '/balances'

    payload = {
        'request': API_VERSION + endpoint,
        'nonce': get_nonce()
    }

    response = _invoke_api(endpoint, payload, keys=input.key_set, pub=False, encode=False)
    response = json.loads(response)
    for balance in response:
       if balance["currency"] == input.currency:
            balance_value = 0

            if balance["currency"] == "BTC":
                balance_value =  market_base.calc_value("btc", balance["available"])

            elif balance["currency"] == "ETH":
                balance_value = market_base.calc_value("eth", balance["available"])

            elif balance["currency"] == "USD":
                balance_value = balance["available"]

            id = balance["currency"] + balance["type"] + input.key_set["public"]
            return market_base.profit_over_time(id, balance_value)

# Features
# =====================================================================================================================>
