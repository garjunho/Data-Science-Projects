import tweepy
from textblob import TextBlob
from dotenv import load_dotenv
import os


load_dotenv()


api_key = os.getenv("TWITTER_API_KEY")
api_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")  


auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

public_tweets = api.search_tweets("Python", count=10)
for tweet in public_tweets:
    print("\nTweet:", tweet.text)
    analysis = TextBlob(tweet.text)
    polarity = analysis.sentiment.polarity
    print("Polarity:", polarity)

    if polarity > 0.5:
        print("Positive tweet detected!")
    elif polarity < 0:
        print("Negative tweet detected!")
    else:
        print("Neutral tweet detected!")

