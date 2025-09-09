from nltk.corpus import wordnet

synset = wordnet.synsets("knivess")
print(synset)
print("lenght of sysnset " , len(synset))
print(synset[0].definition())

print("\n\n\n\t printing defination associated to each word\n\n")

for word in synset:
    print(word.name() , word.definition())
    print("example: " , word.examples())



    
