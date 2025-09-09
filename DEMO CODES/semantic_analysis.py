from nltk.corpus import wordnet

wOne = wordnet.synset("program.n.01");
wThree = wordnet.synset("program.n.01")
wTwo = wordnet.synset("university.n.01")
print(wOne.wup_similarity(wThree))
