#!/usr/bin/python

import urllib2
import json
import datetime

import csv

FILENAME = "stocktwits.json"  # change as necessary

fin_sector_tickers = ["AAPL", "MSFT", "IBM", "ORCL", "GOOGL", "QCOM", "INTC", "FB", "CSCO", "V", "MA", "EBAY", "HPQ",
                      "EMC", "ACN", "TXN", "YHOO", "ADP", "CRM", "ADBE", "CTSH", "MU", "GLW", "AMAT", "TEL", "INTU",
                      "SNDK", "WDC", "MSI", "STX", "APH", "ADI", "BRCM", "FIS", "FISV", "PAYX", "XRX", "ADS", "CA",
                      "SYMC", "JNPR", "NTAP", "XLNX", "ADSK", "CTXS", "ALTR", "KLAC", "LLTC", "NVDA", "AKAM", "CSC",
                      "EA", "LRCX", "MCHP", "RHT", "HRS", "WU", "FFIV", "FSLR", "TDC", "LSI", "TSS", "VRSN", "FLIR",
                      "JBL", "GOOG"]


def get_tweets(ticker):
    url = "https://api.stocktwits.com/api/2/streams/symbol/{0}.json".format(ticker)
    connection = urllib2.urlopen(url)
    data = connection.read()
    connection.close()
    return json.loads(data)


def get_tweets_list(tickers, outputF):
    ret = {}
    with open(outputF, 'w') as csvfile:
        tweet_writter = csv.writer(csvfile, delimiter=',', quotechar='\"')
        tweet_writter.writerow(['ticker', 'text', 'sentiment'])

        for ticker in tickers:
            print "Getting data for", ticker
            try:
                data = get_tweets(ticker)
                symbol = data['symbol']['symbol']
                msgs = data['messages']
                for i in range(0, len(msgs)):
                    text = data['messages'][i]['body']
                    sentiment = data['messages'][i]['entities']['sentiment']
                    if sentiment == None:
                        sentiment = 'neutral'

                    elif data['messages'][i]['entities']['sentiment']['basic'] == 'Bearish':
                        sentiment = 'neg'
                    elif data['messages'][i]['entities']['sentiment']['basic'] == 'Bullish':
                        sentiment = 'pos'
                    else:
                        sentiment = 'error'

                    tweet_writter.writerow([ticker, text.encode('utf8'), sentiment])

                ret.update({symbol: msgs})
            except Exception as e:
                print e
                print "Error getting", ticker
    return ret


# schema for original and msgs: ticker (key) : msgs (value, list)
def append(original, msgs):
    print "Appending tweets"
    for ticker in msgs.keys():
        if ticker not in original.keys():
            original[ticker] = msgs[ticker]
        else:
            for msg in msgs[ticker]:
                if msg not in original[ticker]:  # check for duplicates
                    original[ticker].append(msg)
    return original


def cull(original, age_limit=26):
    # cull all tweets over age_limit days old
    print "Culling tweets that are more than", age_limit, "days old"
    threshold = datetime.datetime.now() - datetime.timedelta(age_limit)
    result = {}
    for ticker in original.keys():
        result[ticker] = []
        for msg in original[ticker]:
            dt = datetime.datetime.strptime(msg["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            if dt >= threshold:
                result[ticker].append(msg)
    return result


def read_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def write_to_file(filename, d):
    with open(filename, 'w+') as f:
        print "Dumping JSON to", filename
        json.dump(d, f)


if __name__ == "__main__":
    # old = read_from_file(FILENAME)
    new = get_tweets_list(fin_sector_tickers, './StockTwits_outputs/StockTwits_scrapes.csv')


    # new = append(old, new)
    # new = cull(new)
    # write_to_file(FILENAME, new)
