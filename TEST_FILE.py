#TEST file generated by Lucas to test the tweet cleaning
import re

pos= open('../Tweets/train_pos.txt').readlines()
pos_correct = open ('train_pos_correct.txt').readlines ()
neg = open ('../Tweets/train_neg.txt').readlines ()
neg_correct=open('train_neg_correct.txt').readlines()

print(len(pos))
print(len(pos_correct))
print(len(neg))
print(len(neg_correct))
'''
print(pos_correct[0:5])
pos_correct[1]=re.sub('\n','', pos_correct[1])
print(pos_correct[0:5])
'''
#ULTRA BRUTE FORCE mais bon, j'avais pas trop le temps
# il doit y avoir un petit problème avec le code de clément, ou alors c'est moi qui fais de la merde
"""
while '\n' in pos_correct:
    pos_correct.remove('\n')
while '\n' in neg_correct:
    neg_correct.remove('\n')


print(len(pos_correct))
print(len(neg_correct))
with open('train_pos_correct.txt','w', encoding='utf8') as f:
    for tweet in pos_correct:
        f.write(tweet)

with open('train_neg_correct.txt','w', encoding='utf8') as f:
    for tweet in neg_correct:
        f.write(tweet)
"""
