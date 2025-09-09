##from nltk.tokenize import sent_tokenize, word_tokenize
##
##
##sampleWord = "hello this, hello Mr. Ade are you cool? is a cool word. The weather is great and python is awesoml"
##
##print(sent_tokenize(sampleWord))
##print(word_tokenize(sampleWord))
##
##
##for word in word_tokenize(sampleWord):
##    print(word)



##
##from nltk.tokenize import word_tokenize
##from nltk.corpus import stopwords
##
##
##sampleText = "this is the coolest word. and i am in shape"
##
##stopword = list(stopwords.words("english"))
##print("Stopt words \n" , stopword)
##
##tokenWord  = word_tokenize(sampleText)
##
##cleanData = [word for word in tokenWord if word not in stopword]
##print(cleanData)


##from nltk.stem import PorterStemmer 
##from nltk.tokenize import word_tokenize
##
##ps = PorterStemmer()
##
##word_list = ["python" , "pythoning" , "pythonly" , "pythoned"]
##
##print(word_list) 
##for w in word_list :
##    print(ps.stem(w))

##  THE PARTH OF SPEECH TAGGING 

##from nltk.corpus import wordnet

##ss = wordnet.synsets("man")
##print(ss)
##print(ss[0].definition())
##
##for s in ss:
##    for l in s.lemmas():
##        print(f"lamma name : {l}")
##



##
##ss = wordnet.synset("man.n.01")
##w2 = wordnet.synset("men.n.01")
##print(ss.wup_similarity(w2))
##


name = "akintunde"

if name is "akintunde":
    print("this is correct")
else:
    print("this is wrong")















