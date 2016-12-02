# for docs see http://andybromberg.com/sentiment-analysis-python/

import pandas as pd

import re, math, collections, itertools, os
import pickle

import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, maxent
from nltk.corpus import stopwords

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from nltk.metrics import precision, recall

DATA_FLAG='kooshas_data'
POLARITY_DATA_DIR = os.path.join('./sentiment_data')
testing_data = './sentiment_data/IT_testing_set.xlsx'

if DATA_FLAG=='kooshas_data':
    RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'IT_training_pos.txt')
    RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'IT_training_neg.txt')
else:
    RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
    RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')


stopWords = stopwords.words('english')
stopWords.append('AT_USER')
stopWords.append('URL')



def pickle_cls(cls, fname):
    f = open(fname + '.pickle', 'wb')
    pickle.dump(cls, f)
    f.close()

def get_performance(clf_sel, train_features, test_features):
    ref_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)
    classification_error=False

    clf = SklearnClassifier(clf_sel)
    try:
        classifier = clf.train(train_features)
    except:
        classification_error=True
        #print (str(clf_sel.__class__),'NA')


    if str(clf_sel.__class__) == "<class 'sklearn.naive_bayes.MultinomialNB'>":
        pickle_cls(classifier, 'MultinomialNB')

    # print(str(clf_sel), 'accuracy:'(nltk.classify.accuracy(classifier, test_features)) * 100)

    if not classification_error:
        clf_acc = nltk.classify.accuracy(classifier, test_features)

        for i, (features, label) in enumerate(test_features):
            ref_set[label].add(i)
            predicted = classifier.classify(features)
            test_set[predicted].add(i)

        pos_precision = precision(ref_set['pos'], test_set['pos'])
        pos_recall = recall(ref_set['pos'], test_set['pos'])
        neg_precision = precision(ref_set['neg'], test_set['neg'])
        neg_recall = recall(ref_set['neg'], test_set['neg'])

        print(
        "{0},{1},{2},{3},{4},{5}".format(clf_sel.__class__, clf_acc, pos_precision, pos_recall, neg_precision, neg_recall))

def get_features(fl_path, no_classes, feat_select):
# creating features from testing dataset
    posFeat = []
    negFeat = []
    neutFeat = []
    df_testing = pd.read_excel(fl_path, header=0)
    for ind in df_testing.index:
        sentiment_code = df_testing.iloc[ind,0]
        i = clean_text(df_testing.iloc[ind,1]) # gets the text in each row

        if sentiment_code==1:
            posW = [w for w in i.lower().split() if w not in stopWords]
            posW = [feat_select(posW), 'pos']
            posFeat.append(posW)
        elif sentiment_code==3:
            negW = [w for w in i.lower().split() if w not in stopWords]
            negW = [feat_select(negW), 'neg']
            negFeat.append(negW)
        else:
            neutW = [w for w in i.lower().split() if w not in stopWords]
            neutW = [feat_select(neutW), 'neutral']
            neutFeat.append(neutW)

    return negFeat,posFeat, neutFeat


# this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select, classifier_sel):
    posFeatures = []
    negFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list

    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            i = clean_text(i)
            posWords = [w for w in i.lower().split() if w not in stopWords]

            # posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            i = clean_text(i)
            negWords = [w for w in i.lower().split() if w not in stopWords]

            # negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

            # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    if (DATA_FLAG == 'kooshas_data'):

        trainFeatures = posFeatures+negFeatures

        neg_test, pos_test, neut_test = get_features(testing_data,no_classes=2,feat_select=feature_select)
        testFeatures = pos_test + neg_test # for three class data add neutFeat

    else:
        posCutoff = int(math.floor(len(posFeatures) * 4 / 5))
        negCutoff = int(math.floor(len(negFeatures) * 4 / 5))
        trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
        testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

        aa,bb, cc = get_features(testing_data,no_classes=2,feat_select=feature_select)
        testFeatures = aa+bb

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # trains a Classifier
    if classifier_sel == 'NB':
        classifier = NaiveBayesClassifier.train(trainFeatures)
        # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = classifier.classify(features)
            testSets[predicted].add(i)

            # prints metrics to show how well the feature selection did
        print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
        print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))

        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = classifier.classify(features)
            testSets[predicted].add(i)

        print ('pos precision:', precision(referenceSets['pos'], testSets['pos']))
        print ('pos recall:', recall(referenceSets['pos'], testSets['pos']))
        print ('neg precision:', precision(referenceSets['neg'], testSets['neg']))
        print ('neg recall:', recall(referenceSets['neg'], testSets['neg']))


        classifier.show_most_informative_features(10)

    elif classifier_sel == 'MaxEnt':
        get_performance(LogisticRegression(), trainFeatures, testFeatures)

    elif classifier_sel == 'all_classifiers':
        get_performance(MultinomialNB(), trainFeatures, testFeatures)
        get_performance(BernoulliNB(), trainFeatures, testFeatures)
        get_performance(LogisticRegression(), trainFeatures, testFeatures)
        get_performance(SGDClassifier(), trainFeatures, testFeatures)
        get_performance(SVC(), trainFeatures, testFeatures)
        get_performance(LinearSVC(), trainFeatures, testFeatures)
        get_performance(NuSVC(kernel='rbf',nu=1), trainFeatures, testFeatures)



    elif classifier_sel == 'SVM':  # use SVM
        SVC_classifier = SklearnClassifier(SVC())
        classifier = SVC_classifier.train(trainFeatures)
        print("SVC_classifier accuracy:",
              (nltk.classify.accuracy(classifier, testFeatures)) * 100)

        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = classifier.classify(features)
            testSets[predicted].add(i)

        get_performance(classifier, referenceSets, testSets)


# creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


# scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            i = clean_text(i)
            posWord = [w for w in i.lower().split() if w not in stopWords]

            # posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)

    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            i = clean_text(i)
            negWord = [w for w in i.lower().split() if w not in stopWords]

            # negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)

    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

        # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


def get_part_speech():
    pass


def getStopWordList(stopWordListFileName):
    # read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


def clean_text(tweet):  # this also removes stop words

    # Convert to lower case
    tweet = tweet.lower()
    # Convert to anything after $ to CASHTAG
    tweet = re.sub('\$[^\s]+', 'CASHTAG', tweet)
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')

    return tweet


def get_tweets(df, cashtag=None, date_in=None):
    tweets = []
    if cashtag == None and date_in == None:
        for row in range(len(df)):
            tweets.append(df.iloc[row, 1])

    elif date_in == None:  # return all tweets in the dataset for a given ticker
        df_ret = df[df['text'].str.contains("\\" + cashtag)]  # the "\" is to scape $
        tweets = df_ret['text'].tolist()

    elif date_in:
        tmp_df = df.loc[str(date_in)]
        df_ret = tmp_df[tmp_df['text'].str.contains("\\" + cashtag)]  # the "\" is to scape $
        tweets = df_ret['text'].tolist()
    else:
        print ("error - ticker and date:", cashtag, date_in)

    return tweets


def get_aggregate_sentiment(tweets_lst, cls):
    classifier = cls
    # convert tweets_lst to test_features

    posFeatures = []
    for tw in tweets_lst:
        posWords = [w for w in tw.lower().split() if w not in stopWords]
        posWords = [make_full_dict(posWords), 'no label']
        posFeatures.append(posWords)

    test_set = collections.defaultdict(set)
    for i, (features, label) in enumerate(posFeatures):
        # ref_set[label].add(i)
        predicted = classifier.classify(features)
        test_set[predicted].add(i)

    if (len(test_set['pos']) + len(test_set['neg'])) != 0:  # check for division by zero
        print (len(test_set['pos']),len(test_set['neg']))
        sentiment = float(len(test_set['pos']) - len(test_set['neg'])) / (len(test_set['pos']) + len(test_set['neg']))
    else:
        sentiment = 0

    return sentiment


def compute_sentiments(start, end, fname, tickers):

    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_tweets = pd.read_csv(fname, header=0, parse_dates=['created_at'], index_col=1)
    df_tweets.index = df_tweets.index.map(lambda t: t.strftime('%Y-%m-%d'))  # remove time from index

    f = open('MultinomialNB.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()

    # tweets_all = get_tweets(df_tweets)
    Bdays = pd.bdate_range(start, end)
    Bdays = pd.DatetimeIndex(Bdays).normalize()  # remove hours from datetime

    tickers = [s.strip('$') for s in tickers]
    df_sentiments = pd.DataFrame(index=Bdays, columns=tickers)

    for ind in df_sentiments.index:  # for each day
        for ticker in tickers:  # calculate the sentiment of each tweet and aggregate

            tweets = get_tweets(df_tweets, ticker, ind.date())

            sent = get_aggregate_sentiment(tweets, classifier)
            print ticker, sent
            df_sentiments.loc[ind, ticker] = sent

    return df_sentiments


def get_tweets_stats(fname, tickers):
    df_tweets = pd.read_csv(fname, header=0)
    df_tweet_stats = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        tweets = get_tweets(df_tweets, ticker)
        df_tweet_stats.loc[1, ticker] = len(tweets)

    return df_tweet_stats

def get_stats(inp, ticks, outp):
    stats_df = get_tweets_stats(inp, ticks)
    stats_df.to_csv(outp, sep=',', encoding='utf-8')



if __name__ == '__main__':

    oil_file = '/Users/kooshag/Google Drive/code/data_sentiment/Oil_all_tweets.csv'
    oil_tickers = ["$XOM", "$CVX", "$SLB", "$COP", "$OXY", "$EOG", "$HAL", "$APC",
                   "$PSX", "$APA", "$NOV", "$KMI", "$BHI", "$VLO", "$DVN", "$HES", "$MRO", "$WMB", "$MPC", "$PXD",
                   "$NBL", "$SE", "$CHK", "$EQT", "$SWN", "$COG", "$RIG", "$CAM", "$FTI", "$RRC", "$ESV", "$OKE", "$HP",
                   "$MUR", "$CNX", "$NE", "$TSO", "$DO", "$NBR", "$DNR", "$BTU", "$QEP", "$NFX", "$RDC"]

    info_tech_file = '/Users/kooshag/Google Drive/code/data_sentiment/twitter_training_combined_SP_financials.csv'
    info_tech_tickers = ["$AAPL", "$MSFT", "$IBM", "$ORCL", "$GOOGL", "$QCOM", "$INTC", "$FB", "$CSCO", "$V", "$MA",
                         "$EBAY", "$HPQ", "$EMC", "$ACN", "$TXN", "$YHOO", "$ADP", "$CRM", "$ADBE", "$CTSH", "$MU",
                         "$GLW", "$AMAT", "$TEL", "$INTU", "$SNDK", "$WDC", "$MSI", "$STX", "$APH", "$ADI", "$BRCM",
                         "$FIS", "$FISV", "$PAYX", "$XRX", "$ADS", "$CA", "$SYMC", "$JNPR", "$NTAP", "$XLNX", "$ADSK",
                         "$CTXS", "$ALTR", "$KLAC", "$LLTC", "$NVDA", "$AKAM", "$CSC", "$EA", "$LRCX", "$MCHP", "$RHT",
                         "$HRS", "$WU", "$FFIV", "$FSLR", "$TDC", "$LSI", "$TSS", "$VRSN", "$FLIR", "$JBL", "$GOOG"]


    # oil_tickers = ["$XOM", "$CVX"]  # this is for testing

    df_sentiments = compute_sentiments('2016-06-22', '2016-07-27', oil_file, oil_tickers)
    #k get_stats(oil_file, oil_tickers, 'temp_del.csv')


    # tries using all words as the feature selection mechanism
    print 'using all words as features'
    classifier = 'all_classifiers'

    print ("classifier, accuracy, pos precision, pos recall, neg precision, neg recall")

    evaluate_features(make_full_dict, classifier)
    # finds word scores
    word_scores = create_word_scores()

    # numbers of features to select
    numbers_to_test = [10, 100, 1000, 10000, 15000]

    # tries the best_word_features mechanism with each of the numbers_to_test of features

    for num in numbers_to_test:
        print 'evaluating best %d word features' % (num)
        best_words = find_best_words(word_scores, num)
        evaluate_features(best_word_features, classifier)

        # trying different parts of speech canot improve the classifier much

    #### calculating sentiments for each stock

    ## get twitter stats (oil_sector)
    ## get twitter stats (finance_sector)


    ## get anomalies dataframe ('2016-06-22', '2016-07-22')
    #




def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    for word in words:
        stripped = word.strip(delims).lower()
        result.append("not_" + stripped if negation else stripped)

        if any(neg in word for neg in frozenset(["not", "n't", "no"])):
            negation = not negation

        if any(c in word for c in delims):
            negation = False
    return result
