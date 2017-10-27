from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from omni.interfaces.util import encode_response

def search(input):

    consumer_key = input.key_set["consumer_key"]
    consumer_secret = input.key_set["consumer_secret"]
    access_token = input.key_set["access_token"]
    access_secret = input.key_set["access_secret"]

    oauth = OAuth(access_token, access_secret, consumer_key, consumer_secret)

    twitter = Twitter(auth=oauth)

    tweets = twitter.search.tweets(q=input.term) #result_type='recent', lang='en', count=10

    return encode_response(tweets)