#http://norvig.com/spell-correct.html

import re
import json
from collections import Counter
from helpers import progress

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('words_light.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def SpellCheck (tweets, spellCheckFile = 'spellcheck.txt'):
    
    print ('Spell Checking')
    
    spellcheck = json.loads (open (spellCheckFile).read ())
    
    tweets_words = []
    needs_update = False
    
    for tweet in tweets:
        
        words = tweet.split (' ')
        tweets_words.append (words)
        
        if not needs_update:
            
            for word in words:
                
                if word not in spellcheck:
                    needs_update = True
                    break
                    
    if needs_update:
        
        words = []

        for tweet_words in tweets_words:
            words += tweet_words

        words = set (words)

        for word in progress (words, '\tUpdating Spell Check Dictionary...'):
            if word not in spellcheck:
                spellcheck [word] = correction (word)
                
        json.dump (spellcheck, open (spellCheckFile,'w'))
        
    tweets_corrected = []
        
    for tweet_words in progress (tweets_words, '\tSpell Checking Tweets...'):
        
        tweet_corrected = []
        
        for word in tweet_words:
            if len (word) > 0:
                tweet_corrected.append (spellcheck [word])
            
        tweets_corrected.append (' '.join (tweet_corrected))
        
    print ('Terminated')
        
    return tweets_corrected