import numpy as np

def load_lexicons():

    lexicon = np.genfromtxt('Lexicons/unigrams-pmilexicon.txt', comments=None,\
                             dtype=[('mystring','S10'),('myfloat','f8'),('myint','i8'),('myint2','i8')],delimiter="\t")
    words = []
    for index,item in enumerate(lexicon):
        words.append(item[0].decode('UTF-8'))

    return lexicon,words

def score_lexicons(lexicon,words_lexicon, positive_tweets, negative_tweets):

    nbrfeatures=3 #to be defined later

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
                feature_repr[0]+=score_word
                if(score_word>0):
                    feature_repr[1]+=score_word*numPositive
                    feature_repr[2] +=1
                else:
                    feature_repr[1]+=score_word*numNegative
        cpt = cpt+1
        percentage = cpt/len(positive_tweets)
        if (percentage > cpt_perc*0.1):
            print("Percentage of positive tweets treated: ", percentage)
            cpt_perc += 1

        positive_tweets_lexicon_features[index] = feature_repr

    negative_tweets_lexicon_features = np.zeros((len(negative_tweets), nbrfeatures))
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
                feature_repr[0] += score_word
                if (score_word > 0):
                    feature_repr[1] += score_word * numPositive
                    feature_repr[2] += 1
                else:
                    feature_repr[1] += score_word * numNegative

        negative_tweets_lexicon_features[index] = feature_repr

    return positive_tweets_lexicon_features,negative_tweets_lexicon_features


