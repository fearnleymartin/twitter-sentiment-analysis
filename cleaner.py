from nltk.corpus import stopwords
from helpers import progress
from spellcheck import SpellCheck
import re

def getFilterList ():
    
    return ['lower', 'collapse spaces', 'stop-words', 'spell check', 'remove duplicates', 'remove empty', 'remove numbers', 'remove non-char', 'remove <substring>']

def processTweets (tweets, sequence, export = ''):
    
    """
    Some processing taken from: https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
    :param tweets:
    :return:
    """
    
    filterList = getFilterList ()
    filters = []
    
    for filt in sequence:
        filters.append (filt.lower ())
    
    unprocessed_tweets = tweets [:]
    processed_tweets = tweets [:]
    
    stop_words = None
    
    if 'stop-words' in filters:

        stop_words = set (stopwords.words ('english'))
        stop_words.update (set (['<user>','<url>']))
        
    for filt in filters:
        
        if (filt not in filterList) and (filt [:7] != 'remove '):
            raise ValueError ('Filter ' + filt + ' is not recognized. Please call \'getFilterList ()\' to view the allowed filters.')
        
    for filt in filters:
            
        temp_processed = []
        temp_unprocessed = []
        
        if filt == 'lower':
            
            for tweet in progress (processed_tweets, 'Lowering Tweets...'):
                temp_processed.append (tweet.lower())
                
        elif filt == 'collapse spaces':
            
            for tweet in progress (processed_tweets, 'Collapsing Multiple Spaces...'):
                temp_processed.append (re.sub ('[\s]+', ' ', tweet))
                
        elif filt == 'stop-words':
            
            for i in progress (range (len (processed_tweets)), 'Removing Stop-words...'):
                
                tweet = processed_tweets [i]
                
                new_words = []
                words = tweet.split (' ')
                
                for word in words:
                    if word not in stop_words:
                        new_words.append (word)
                        
                if len (new_words) > 0:
                    temp_processed.append (' '.join (new_words))
                    temp_unprocessed.append (unprocessed_tweets [i])
                    
            unprocessed_tweets = temp_unprocessed
                
        elif filt == 'spell check':
            
            temp_processed = SpellCheck (processed_tweets)
                
        elif filt == 'remove duplicates':
            
            for i in progress (range (len (processed_tweets)), 'Removing Duplicates...'):
                
                tweet = processed_tweets [i]
                
                if tweet not in temp_processed:
                    temp_processed.append (tweet)
                    temp_unprocessed.append (unprocessed_tweets [i])
                    
            unprocessed_tweets = temp_unprocessed
                
        elif filt == 'remove empty':
            
            for i in progress (range (len (processed_tweets)), 'Removing Empty Tweets...'):
                
                tweet = processed_tweets [i]
                        
                if len (re.sub ('[\s]+', '', tweet)) > 0:
                    temp_processed.append (tweet)
                    temp_unprocessed.append (unprocessed_tweets [i])
                    
            unprocessed_tweets = temp_unprocessed
                
        elif filt == 'remove numbers':
            
            for tweet in progress (processed_tweets, 'Removing Numbers...'):
                temp_processed.append (re.sub('^\d+\s|\s\d+\s|\s\d+$', ' ', tweet))
                
        elif filt == 'remove non-char':
            
            for tweet in progress (processed_tweets, 'Removing Non Characters...'):
                temp_processed.append (re.sub('[!@#$,.:/\&;()+=°*?§µ%£¤¨}{|<>\'"]', '', tweet))
                
        else:
                                       
            subs = re.sub ('\n', 'next line', filt [7:])
            
            for tweet in progress (processed_tweets, 'Removing Substring \'' + subs + '\'...'):
                temp_processed.append (re.sub(subs, '', tweet))
                
        processed_tweets = temp_processed
        
    if len (export) > 0:
        
        with open (export + '_processed.txt', 'w', encoding = 'utf-8') as f:
            f.write ('\n'.join (processed_tweets))
        
        with open (export + '_unprocessed.txt', 'w', encoding = 'utf-8') as f:
            f.write (''.join (unprocessed_tweets))
            
    return unprocessed_tweets, processed_tweets