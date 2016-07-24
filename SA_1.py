# for docs see http://andybromberg.com/sentiment-analysis-python/

import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, maxent

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from nltk.metrics import precision, recall

POLARITY_DATA_DIR = os.path.join('./sentiment_data')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')


def get_performance(clf_sel, train_features, test_features):
    ref_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)

    clf = SklearnClassifier(clf_sel)
    classifier = clf.train(train_features)
    print(str(clf_sel), 'accuracy:'
    (nltk.classify.accuracy(classifier, test_features)) * 100)

    for i, (features, label) in enumerate(test_features):
        ref_set[label].add(i)
        predicted = classifier.classify(features)
        test_set[predicted].add(i)

    print 'pos precision:', precision(ref_set['pos'], test_set['pos'])
    print 'pos recall:', recall(ref_set['pos'], test_set['pos'])
    print 'neg precision:', precision(ref_set['neg'], test_set['neg'])
    print 'neg recall:', recall(ref_set['neg'], test_set['neg'])


# this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select, classifier_sel):
    posFeatures = []
    negFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

            # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 4 / 5))
    negCutoff = int(math.floor(len(negFeatures) * 4 / 5))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

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
        print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
        print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)

        get_performance(classifier, referenceSets, testSets)

        classifier.show_most_informative_features(10)

    elif classifier_sel == 'MaxEnt':
        pass

    elif classifier_sel == 'other_classifiers':

        get_performance(MultinomialNB(), trainFeatures, testFeatures)
        get_performance(BernoulliNB(), trainFeatures, testFeatures)
        get_performance(LogisticRegression(), trainFeatures, testFeatures)
        get_performance(SGDClassifier(), trainFeatures, testFeatures)
        get_performance(SVC(), trainFeatures, testFeatures)
        get_performance(LinearSVC(), trainFeatures, testFeatures)
        get_performance(NuSVC(), trainFeatures, testFeatures)



    else:  # use SVM
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
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
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


if __name__ == '__main__':
    # tries using all words as the feature selection mechanism
    print 'using all words as features'

    classifier = 'other_classifiers'

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


        # try with different parts of speech
