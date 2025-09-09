##from pattern.en import parse
##from pattern.en import pprint
##
##pprint(parse("I drove my car to the hospital yersterday"
##             , relations=True, lemmata=True))


##from pattern.en import parse
##from pattern.en import pprint
##
##try:
##    print(parse('I drove my car to the hospital yesterday'))
##except RuntimeError as ex:
##    print("run to some an error")


##  PLURALIZAING AND SINGULARIZATION OF TOKENS

from pattern.en import singularize
from pattern.en import pluralize ;
from pprint import pprint
from pattern.en import superlative , comparative
from pattern.en import sentiment
from pattern.en import parse, Sentence, modality
from pattern.en import suggest

words = ["boy" , "boys" , "knives" , "knife" , "drove" , "drive"]

dic  = set()

print(dic)

w1 = "boy"
w2 = "boys"

for word in words:
    if word == singularize(word):
        print(f"{word} is singular word")
        dic.add((word , pluralize(word)))
    else:
        print(f"{word} is plural word")
        dic.add(( singularize(word) , word ))

pprint(list(dic))


print(superlative("good"))
print(comparative("good"))


##print(suggest("whitle"))

    

