import urllib

from nltk.twitter import Query, credsfromfile


def get_sentiment_nltk_web(lst):
    twts = lst
    for twt in twts:
        # sentiment analysis
        data = urllib.urlencode({"text": twt})
        u = urllib.urlopen("http://text-processing.com/api/sentiment/", data)
        the_page = u.read()
        print the_page, twt


if __name__ == '__main__':
    oauth = credsfromfile()
    # search a topic
    # tw = Twitter()
    # tw.tweets(keywords='edmonton arena', stream=False, limit=10)

    client = Query(**oauth)
    tweets = client.search_tweets(keywords='covenant care', limit=10)
    tweet = next(tweets)

    tweets_txt = []
    for tweet in tweets:
        tweets_txt.append(tweet['text'].encode('utf-8'))
        print("- " + tweet['text'])

    # sentiment analysis
    get_sentiment_nltk_web(tweets_txt)

    print ("EOP")
