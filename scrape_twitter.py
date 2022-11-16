import tweepy
import datetime
import pandas as pd
from tqdm import tqdm
import time
import json

class TweetCollector(object):
    with open("../api_token.json", "r") as f:
        tokens = json.loads(f.read())

    auth = tweepy.OAuthHandler(tokens['consumer_key'], tokens['consumer_secret'])
    auth.set_access_token(tokens['access_token'], tokens['access_secret'])
    api = tweepy.API(auth)

    hashtags = [
        '#nft',
        '#climatechange',
        '#christmas',
        '#covid',
        '#ukraine'
    ]
    num_tweets = 1800

    def __init__(self):
        today = datetime.datetime.now()
        time_to_the_past = 7
        self.start_date = today - datetime.timedelta(time_to_the_past)
        self.end_date = today - datetime.timedelta(time_to_the_past - 1)
        print(self.start_date)
        print(self.end_date)

    @classmethod
    def get_tweets(self, tag: str, end_date, df) -> pd.DataFrame:
        
        tweets = tweepy.Cursor(self.api.search_tweets,
                                tag,
                                lang='en',
                                until=end_date.date(),
                                tweet_mode='extended').items(self.num_tweets)
        for tweet in tqdm(list(tweets)):
            tweet_id = tweet.id
            create_at = tweet.created_at
            geo = tweet.user.location
            retweet_count = tweet.retweet_count
            like_count = tweet.favorite_count
            hashtags = tweet.entities['hashtags']
            username = tweet.user.screen_name
            following = tweet.user.friends_count
            followers = tweet.user.followers_count
            user_total_tweets = tweet.user.statuses_count
            user_total_likes = tweet.user.favourites_count
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text
            hash_text = []
            for j in range(len(hashtags)):
                hash_text.append(hashtags[j]['text'])

            ith_tweet = [tweet_id, text, create_at, geo,
                        retweet_count, like_count, hash_text,
                        username, following, followers, user_total_tweets, user_total_likes]
            df.loc[len(df)] = ith_tweet
        # df.create_at = pd.to_datetime(df.create_at).dt.tz_localize(None)
        # df = df[(df['create_at'] < end_date) & (df['create_at'] > start_date)]
        print(df)

    def __call__(self):
        df = pd.DataFrame(columns=['id',
                                'text',
                                'create_at',
                                'geo',
                                'retweet_count',
                                'like_count',
                                'hashtags',
                                'username',
                                'following',
                                'followers',
                                'user_total_tweets',
                                'user_likes_count'
                                ])
        for i in range(len(self.hashtags)):
            if i != 0:
                print(datetime.datetime.now(), ":sleep for 15 minutes")
                time.sleep(15*60)
            self.get_tweets(self.hashtags[i], self.end_date, df)
        filename = 'si670finalproject/data/scraped_tweets-' + str(self.start_date) + '-' + str(self.end_date) + '.csv'
        df.set_index(['id']).to_csv(filename)

if __name__ == "__main__":
    collector = TweetCollector()
    collector()
