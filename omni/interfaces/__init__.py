from omni.interfaces.registration import affordance, task, closer
import requests_cache

# Cache
# --------------------------------------------------------------------------------------------------------------------->


# Store
# --------------------------------------------------------------------------------------------------------------------->



# Base
# --------------------------------------------------------------------------------------------------------------------->

task(entry_point='omni.interfaces.base.penalise:penalise_connection_errors')
task(entry_point='omni.interfaces.base.penalise:penalise_connection_error')
task(entry_point='omni.interfaces.base.penalise:penalise_response_size')
task(entry_point='omni.interfaces.base.penalise:step_loss')

affordance(entry_point="omni.interfaces.base.omni:list_affordances")
affordance(entry_point="omni.interfaces.base.omni:list_tasks")

# Blockchain.info
# --------------------------------------------------------------------------------------------------------------------->

for chart in ["total-bitcoins","market-price","market-cap","trade-volume",
                      "blocks-size", "avg-block-size", "n-orphaned-blocks", "n-transactions-per-block", "median-confirmation-time",
                      "bip-9-segwit", "bitcoin-unlimited-share", "nya-support", "hash-rate", "difficulty", "miners-revenue",
                      "transaction-fees", "transaction-fees-usd", "cost-per-transaction-percent", "cost-per-transaction", "n-unique-addresses",
                      "n-transactions", "n-transactions-total", "transactions-per-second", "mempool-count", "mempool-growth", "mempool-size",
                      "utxo-count", "n-transactions-excluding-chains-longer-than-100", "output-volume", "estimated-transaction-volume",
                      "estimated-transaction-volume-usd", "my-wallet-n-users"]:
    affordance(entry_point='omni.interfaces.web.blockchain:get_chart', chart=chart, cached=True, cache_length=84000)

affordance(entry_point='omni.interfaces.web.blockchain:get_ticker')

affordance(entry_point='omni.interfaces.web.blockchain:get_stats')

affordance(entry_point='omni.interfaces.web.blockchain:get_pools')


# Gemini
# --------------------------------------------------------------------------------------------------------------------->

#task(entry_point='omni.interfaces.markets.gemini:step_loss')
affordance(entry_point='omni.interfaces.markets.gemini:get_symbols')

for pair in ["btcusd", "ethusd", "ethbtc"]:

    affordance(entry_point='omni.interfaces.markets.gemini:get_ticker', symbol=pair)

    affordance(entry_point='omni.interfaces.markets.gemini:get_order_book', symbol=pair)

    affordance(entry_point='omni.interfaces.markets.gemini:get_current_auction', symbol=pair)

    affordance(entry_point='omni.interfaces.markets.gemini:get_auction_history', symbol=pair)

for key_set in [
    {"private": "3go1mGK4QSJkpFMdxtadRM6e9NoM", "public": "FdAVXfnhsnGwiEOOlDJY"},  # smithmalcolm46@gmail.com
    {"private": "u8rGPS1AvbWNqreT2U9rT4xAPPk", "public": "QzeR2u1AZuf5S6lXWrfo"},  # bradkyleduncan@gmail.com
    {"private": "26NheKRMDt6q24NFUASYVDYE4KPw", "public": "meMqYdKRQsxZDjOU6MRn"},  # wilnatfor@gmail.com
]:
    for currency in ["BTC", "ETH", "USD"]:
        task(entry_point='omni.interfaces.markets.gemini:profit_over_time', key_set=key_set, currency=currency)

    closer(entry_point='omni.interfaces.markets.gemini:cancel_all_orders', key_set=key_set)

    affordance(entry_point='omni.interfaces.markets.gemini:get_active_orders', key_set=key_set)

    affordance(entry_point='omni.interfaces.markets.gemini:get_order_status', key_set=key_set)

    affordance(entry_point='omni.interfaces.markets.gemini:cancel_session_orders', key_set=key_set)

    affordance(entry_point='omni.interfaces.markets.gemini:cancel_all_orders', key_set=key_set)

    affordance(entry_point='omni.interfaces.markets.gemini:get_balance', key_set=key_set)

    for pair in ["btcusd", "ethusd", "ethbtc"]:

        affordance(entry_point='omni.interfaces.markets.gemini:get_trade_volume', symbol=pair, key_set=key_set)

        affordance(entry_point='omni.interfaces.markets.gemini:get_past_trades', symbol=pair, key_set=key_set)

        for option in ["maker-or-cancel", "immediate-or-cancel", "auction-only", ""]:
            for side in ["buy", "sell"]:
                affordance(entry_point='omni.interfaces.markets.gemini:new_order', symbol=pair, key_set=key_set, options=option, side=side)

# Twitter
# --------------------------------------------------------------------------------------------------------------------->

for key_set in [
        {"consumer_key":"WM1MVD31TRGZBiNbXO54p4Sni", "consumer_secret":"SJzN8rhCSyyzxt55shwpPScX1aEdOWX63q8d8yKC9gNAESsCjO", "access_token":"367687040-UzRMSNmwLjgjDl3CwAS72UbyeSkDOmKTKzmWSK89", "access_secret":"TAAu1ScD4pscR8nHTp3UyXx2T6JyCqOy5vTmcIYMtuGga"},
        {"consumer_key":"qnv3th6JUVEvTDoSHl5YcoYLl", "consumer_secret":"whgIR7PGf4zmlnlBaTQCIVPAudVBk1Tej93N0xW0vIpavjpQ6W", "access_token":"863416332095823872-LywT0OQYLI9smMDVDlTG6S9ygj8TA2x", "access_secret":"t6lYoMidHEumwiypurUArSzMMrn3VNQDXRsfXIANzZBsV"},
        {"consumer_key":"ghlSi83iK8b6Gt5r40Y5PB3Lh", "consumer_secret":"kSJkkeB3ggT0oErZ7xPyLOBWOOQytgmJPXao9pfhEnLXWXfyfn", "access_token":"367687040-P0WS0qnFihACmKdJNHo4m2tVy6LbUakZsLD7mse1", "access_secret":"Z30IzyTEKo4Tedk4fTazhJsz4YihtqY7NkKwoPAHGARYv"},
        {"consumer_key":"OVxT26UmJtdMLlJFY0vNDchRG", "consumer_secret":"c6iGLYINqU93qehJ9v84M1hVNXtmQGxtz6Pjs7aLO9X2aQh8aR", "access_token":"860540303421448193-RkErbZqOLToJQtxwk6oV5UhMEb7C9Uq", "access_secret":"VdOlI89dDaLYhCGXYZycWDAuGbVnklCbXyejpiSkziObK"},
        {"consumer_key":"ONoSdnInA2MHk5swW1FiGHBNl", "consumer_secret":"5pEwoWz97OE8BVTogM1nxtHfdkSjPkZkzAektGt8QAHGojkNuo", "access_token":"804937531749974016-ChkHvQwqOtsX1pKPnQRa4MqdGngXMTA", "access_secret":"OG7dxmYBD63749upwQRvSeJlafyJP6gW7SRWpJdWpMpYb"},
        {"consumer_key":"kIFURn1JdN2k1w0WUgENYz8w1", "consumer_secret":"8djbRqNF9Pg7J7Ofjv2tdXbObEWazrr4xt8FxWKMiev8qwhexb", "access_token":"860540303421448193-XJt9pKmqUye0yPXkSlm49LEtFHQ7lQW", "access_secret":"EiiuRn97OKEt4KXW7SQfmXvVjkR1Ec3ZHmTi1luRIUmKI"},
        {"consumer_key":"Zv9zY9KWONeU3qKbShxq9Oat9", "consumer_secret":"Uotuon99K26RCxtUkgB7KIBMjIKLaEQ8CFoCikCkqmDAstJCBX", "access_token":"860540303421448193-IgRs21k3HerMXA7eU97S90BZ3C460kF", "access_secret":"ab1WSco1X6jzmhNRNAnD1ieF7wmYGAKwK3ZbixwgyjuON"},
        {"consumer_key":"SpPvOhlmtNM3vDzK67XxT6r4A", "consumer_secret":"QvkZjhp4YnmpnE2bZpvsHe0HYwbuNonMLgplgQKbV3NUOfaOHH", "access_token":"804937531749974016-BynNs2n2PaK2VTvg0YlK92aBcArd9dr", "access_secret":"wvQ6jiey1T6ORhIwHZm3rMJi5UdRQTQgijHtqRUy9Xn7L"},
        {"consumer_key":"gtyPPsUvl94WPsevhBMm1xGQ2", "consumer_secret":"lgVN2CFlsXdVthgt4ZdVv1ExQNwCX2zdqv2rWkZ7ccpjkbuVG5", "access_token":"863416332095823872-MyeGXCHVAWwgQG93rCtIKpxS7444bou", "access_secret":"cDr0TZ7dHzx0HVzCdpLfkivyRjl2dDvYjXHkaXOxa2vp6"},
        {"consumer_key":"Zv9zY9KWONeU3qKbShxq9Oat9", "consumer_secret":"Uotuon99K26RCxtUkgB7KIBMjIKLaEQ8CFoCikCkqmDAstJCBX", "access_token":"860540303421448193-IgRs21k3HerMXA7eU97S90BZ3C460kF", "access_secret":"ab1WSco1X6jzmhNRNAnD1ieF7wmYGAKwK3ZbixwgyjuON"},
]:
    for term in ["bitcoin", "btc", "ether", "ethereum", "eth"]:
        affordance(entry_point='omni.interfaces.twitter.twitter:search', key_set=key_set, term=term)


# Quora
# --------------------------------------------------------------------------------------------------------------------->
for term in ["FRED/GDP", "BNC3/GWA_BTC", "BNC3/GWA_LTC",
             "USTREASURY/REALLONGTERM", "USTREASURY/REALYIELD", "USTREASURY/BILLRATES", "USTREASURY/YIELD", "USTREASURY/LONGTERMRATES", "USTREASURY/HQMYC",
             "USTREASURY/MATDIS", "USTREASURY/AVMAT", "USTREASURY/TNMBOR", "USTREASURY/TMBOR", "USTREASURY/MKTDM", "USTREASURY/BRDNM"]:
    affordance(entry_point='omni.interfaces.quandl.quandl:search', term=term)

# Etherscan
# --------------------------------------------------------------------------------------------------------------------->
for chart in ["tx","address","etherprice","marketcap",
              "ethersupplygrowth", "hashrate", "difficulty", "pendingtx", "blocks",
              "uncles", "blocksize", "blocktime", "gasprice", "gaslimit", "gasused",
              "ethersupply", "chaindatasizefull", "chaindatasizefast", "ens-affordance"]:
    affordance(entry_point='omni.interfaces.web.etherscan:get_chart', chart=chart)

# Coinmarketcap
# --------------------------------------------------------------------------------------------------------------------->
for convert in ["AUD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR", "INR", "JPY", "KRW", "MXN", "RUB"]:
    affordance(entry_point='omni.interfaces.web.coinmarketcap:get_all_tickers', cache=True, cache_length=300, convert=convert)

for convert in ["AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR",
                "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN", "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR"]:
    affordance(entry_point='omni.interfaces.web.coinmarketcap:get_stats', convert=convert)





