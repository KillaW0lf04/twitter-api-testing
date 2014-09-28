import sys
import json
import pandas as pd

from textblob.sentiments import NaiveBayesAnalyzer
from birdy.twitter import UserClient
from collections import Counter


def main():
    with open('secret.json', 'r') as fp:
        credentials = json.load(fp)

    client = UserClient(**credentials)

    response = client.api.search.tweets.get(q=sys.argv[1], count=150)

    # Need to break down hierarchy (example 'user' column)
    tweets = pd.DataFrame(response.data['statuses'])

    # Doesnt seem to the best classifier - its quite shit actually
    # Maybe movie reviews for a training corpus isn't the most ideal
    # representation of tweets?
    # A lot of false positives and a few false negatives
    print('Training classifier...')
    classifier = NaiveBayesAnalyzer()
    classifier.train()  # Train on a Movie Review Corpus

    print('Performing Sentiment Analysis...')
    counter = Counter()

    for text in tweets['text']:
        result = classifier.analyze(text)

        counter[result.classification] += 1
        print '%s: %s' % (result.classification, text)

    print 'Total: ', counter

if __name__ == '__main__':
    main()
