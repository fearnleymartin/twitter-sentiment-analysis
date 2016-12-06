import numpy as np
from feature_repr import load_tweets

def load_lexicons():

    lexicon = np.genfromtxt('Lexicons/unigrams-pmilexicon.txt', comments=None,\
                             dtype=[('mystring','S10'),('myfloat','f8'),('myint','i8'),('myint2','i8')],delimiter="\t")
    words = []
    for index,item in enumerate(lexicon):
        words.append(item[0].decode('UTF-8'))

    return lexicon,words

def score_lexicons(lexicon,words_lexicon, positive_tweets, negative_tweets):

    nbrfeatures=6 #to be defined later

    positive_tweets_lexicon_features = np.zeros((len(positive_tweets), nbrfeatures))

    cpt = 0
    cpt_perc=0
    for index, tweet in enumerate(positive_tweets):
        words = tweet.split(' ')
        feature_repr = np.zeros(nbrfeatures)
        for word in words:
            if word in words_lexicon:
                index_word = words_lexicon.index(word)
                score_word = lexicon[index_word][1]
                numPositive = lexicon[index_word][2]
                numNegative = lexicon[index_word][3]

                #feature construction
                feature_repr[0]+=score_word#total score
                if(score_word>0):
                    feature_repr[1]+=score_word*numPositive
                    feature_repr[2] +=1
                else:
                    feature_repr[1]+=score_word*numNegative
                    feature_repr[3] +=1

                if(abs(score_word)>feature_repr[4]):
                    feature_repr[4]=score_word #maximal score in abs value

                feature_repr[5] += numPositive-numNegative

        cpt = cpt+1
        percentage = cpt/len(positive_tweets)
        if (percentage > cpt_perc*0.1):
            print("Percentage of positive tweets treated: ", percentage)
            cpt_perc += 1
        positive_tweets_lexicon_features[index] = feature_repr

    negative_tweets_lexicon_features = np.zeros((len(negative_tweets), nbrfeatures))

    cpt = 0
    cpt_perc = 0
    for index, tweet in enumerate(negative_tweets):
        words = tweet.split(' ')
        feature_repr = np.zeros(nbrfeatures)
        for word in words:
            if word in words_lexicon:
                index_word = words_lexicon.index(word)
                score_word = lexicon[index_word][1]
                numPositive = lexicon[index_word][2]
                numNegative = lexicon[index_word][3]

                # feature construction
                feature_repr[0] += score_word  # total score
                if (score_word > 0):
                    feature_repr[1] += score_word * numPositive
                    feature_repr[2] += 1
                else:
                    feature_repr[1] += score_word * numNegative
                    feature_repr[3] += 1

                if (abs(score_word) > feature_repr[4]):
                    feature_repr[4] = score_word  # maximal score in abs value

                feature_repr[5] += numPositive - numNegative

        cpt = cpt + 1
        percentage = cpt / len(negative_tweets)
        if (percentage > cpt_perc * 0.1):
            print("Percentage of negative tweets treated: ", percentage)
            cpt_perc += 1
        negative_tweets_lexicon_features[index] = feature_repr

    np.save('pos_tweets_lexicon_features',positive_tweets_lexicon_features)
    np.save('neg_tweets_lexicon_features', negative_tweets_lexicon_features)
    return positive_tweets_lexicon_features,negative_tweets_lexicon_features

if __name__ == '__main__':
    positive_tweets, negative_tweets = load_tweets()
    print("Tweets loaded")
    lexicon,words_lexicon=load_lexicons()
    print("Lexicon and words loaded")
    positive_tweets_lexicon_repr = score_lexicons(lexicon, words_lexicon, positive_tweets, negative_tweets)
    print("Features from lexicon built")
