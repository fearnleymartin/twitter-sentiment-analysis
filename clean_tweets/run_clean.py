from feature_repr import load_tweets
import re
from nltk.corpus import stopwords

def process_tweets(tweets):
    """
    Some processing taken from: https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
    :param tweets:
    :return:
    """

    stop_words = set(stopwords.words('english'))
    stop_words.update(set(['AT_USER', 'URL', '<user>','<url>']))
    stop_characters = set(['1','2','3','4','5','6','7','8','9',])
    processed_tweets = set([])  # set to avoid duplicates
    for tweet in tweets:
        tweet = tweet.lower()  # remove capitals
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
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
        words = tweet.split(' ')  # split into words
        # define and remove stopwords (i.e. common words)
        processed_words = []
        for word in words:
            if word not in stop_words:
                if len(stop_characters & set(word)) > 0:  # don't add words that contain stop characters
                    break
                word = re.sub('[!@#$,.:/\&;]', '', word)  # remove al punctuation
                processed_words.append(word)
        # words = [w.strip('\'"?,.') for w in words if w not in stop_words]
        if len(processed_words) > 0:
            processed_tweets.add(' '.join(processed_words))
    return list(processed_tweets)

def process_tweets_to_file(positive_tweets, negative_tweets):
    """ Create cleaned version of both sets of tweets and save to file"""
    processed_pos_tweets = process_tweets(positive_tweets)
    processed_neg_tweets = process_tweets(negative_tweets)
    with open('train_neg_full_cleaned.txt', 'w', encoding='utf-8') as f:
        for tweet in processed_neg_tweets:
            f.write(tweet + '\n')
        f.writelines(processed_neg_tweets)
    with open('train_pos_full_cleaned.txt', 'w', encoding='utf-8') as f:
        for tweet in processed_pos_tweets:
            f.write(tweet + '\n')

def combine():
    with open('train_neg_full_cleaned.txt', 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
    with open('train_pos_full_cleaned.txt', 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open('train_full_cleaned.txt', 'w', encoding='utf-8') as f:
        tweets = pos_tweets + neg_tweets
        for tweet in tweets:
            f.write(tweet)

if __name__ == "__main__":
    positive_tweets, negative_tweets, test_tweets = load_tweets(full=True, cleaned=False)
    process_tweets_to_file(positive_tweets, negative_tweets)
    # combine()

