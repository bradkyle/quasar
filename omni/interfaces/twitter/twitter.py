from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from omni.interfaces.processing import process

def search(input):

    consumer_key = input.key_set["consumer_key"]
    consumer_secret = input.key_set["consumer_secret"]
    access_token = input.key_set["access_token"]
    access_secret = input.key_set["access_secret"]

    count = input.args[0] * 50

    oauth = OAuth(access_token, access_secret, consumer_key, consumer_secret)

    twitter = Twitter(auth=oauth)

    tweets = twitter.search.tweets(q=input.term, result_type='recent' , count=count)

    return process(tweets)