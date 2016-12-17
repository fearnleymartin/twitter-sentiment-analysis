Les notebooks sont là pour montrer comment utiliser les différents scripts

Word2Vec demonstration contient aussi l'envoi Kaggle (donc un score de 0.81820)

Pour le clean des tweets, c'est encore expérimental, donc voyez les scripts spellcheck.py et cleaner.py.
Mais grosso modo, SpellCheck s'utilise sur une liste de tweets et les corrige, et processTweets
(dans cleaner.py) s'appelle aussi sur une liste de tweets et demande une liste de filtres à appliquer.
Pour choper la liste des filtres dispos, faut appeler getFilterList. Pour donner une liste de filtre,
l'ordre est important. Par exemple, si vous voulez mettre en lowercase, puis spellcheck, puis retirer
les duplicatas, faut donner ['lower', 'spell check', 'remove duplicates'].