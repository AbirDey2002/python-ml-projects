import nltk
from nltk.corpus import twitter_samples
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def tweets() -> (list, list):
    nltk.download('twitter_samples')

    all_pos_tweets = twitter_samples.strings('positive_tweets.json')
    all_neg_tweets = twitter_samples.strings('negative_tweets.json')

    nltk.download('stopwords')
    stopwords_english = stopwords.words('english')

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    stemmer = PorterStemmer()

    pos_processed_tweets = []
    neg_processed_tweets = []

    for tweet in all_pos_tweets:
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet_tokens = tokenizer.tokenize(tweet)
        clean_tokens = []
        for word in tweet_tokens:
            if word not in stopwords_english and word not in string.punctuation:
                stem_word = stemmer.stem(word)
                clean_tokens.append(stem_word)
        pos_processed_tweets.append(clean_tokens)

    for tweet in all_neg_tweets:
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet_tokens = tokenizer.tokenize(tweet)
        clean_tokens = []
        for word in tweet_tokens:
            if word not in stopwords_english and word not in string.punctuation:
                stem_word = stemmer.stem(word)
                clean_tokens.append(stem_word)
        neg_processed_tweets.append(clean_tokens)

    return pos_processed_tweets, neg_processed_tweets
