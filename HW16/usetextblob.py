import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

data = pd.read_csv("IMDB Dataset.csv")

text = data.iloc[1,0]

print(text)

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')


print(text)

doc = nlp(text)
print('Polarity:')
print(doc._.blob.polarity)                            # Polarity: -0.125
print('Subjectivity:')
print(doc._.blob.subjectivity)                        # Subjectivity: 0.9
print('Sentiment assessments:')
print(doc._.blob.sentiment_assessments.assessments)   # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
print('ngrams:')
print(doc._.blob.ngrams())
print('result')
print(data.iloc[0,1])