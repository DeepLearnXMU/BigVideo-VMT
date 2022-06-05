import spacy
nlp = spacy.load("zh_core_web_sm")
doc = nlp("庆祝祖国生日快乐,哦耶")
print(doc.text)