from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

sample_word = "this is the coolest word ever and it is ok for me"


token_word = word_tokenize(sample_word)
stopWord = set(stopwords.words("english"))
print(token_word)

# list comperssion 
processedWord = [st for st in token_word if st not in stopWord]
print(processedWord)

