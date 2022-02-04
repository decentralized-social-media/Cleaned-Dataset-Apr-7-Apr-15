#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 23:46:59 2021

@author: catherine
"""
import pandas as pd
continentname = ['Africa','Asia','Australia','Europe','North America','South America']
dataset = pd.read_csv("/Users/catherine/Desktop/output1.csv")
comments = dataset['body']
continents = dataset['Continent']
data = {}
import re
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

for each in range(0,len(dataset)):
    commenteach = re.sub(r"\S*https?:\S*", "", str(comments[each]))
    commenteach = cleanhtml(commenteach)
    if continents[each] in data:
        data[continents[each]].append(commenteach)
    else:
        data[continents[each]] = [commenteach]
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text


data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['body']
data_df = data_df.sort_index()
import string
def clean_text_round1(text): 
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_df.body.apply(round1))

# Apply a second round of cleaning
def clean_text_round2(text):
    # Type your code here
    # 2 points for cleaning the data (full points for doing 2/3 below)
    # some of the things to try:
    # (1) remove any of the puntuations
    text = re.sub('[‘’“”…]', '', text)
    # (2) remove new line characters
    text = re.sub('\n',' ',text)
    text = re.sub('ing','',text)
    return text

# 2 points for cleaning the data
round2 = lambda x: clean_text_round2(x)
data_clean2 = pd.DataFrame(data_clean.body.apply(round2))


from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl


cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.body)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index


'''
data_dtm.to_pickle("/Users/catherine/Desktop/dtm.pkl")
with open("dtm.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'file.csv')
'''


import numpy as np

# This is the count vectorizer's full signature, with various parameter values
# The 2 points are awarded for demonstrating the use of ngram_range, max_df, and min_df
CountVectorizer(input='content', encoding='utf-8', decode_error='strict', 
                strip_accents=None, lowercase=True, preprocessor=None, 
                tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', 
                ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, 
                max_features=None, vocabulary=None, binary=False, dtype=np.int64)

# Here is a some sample experimentation we would like to see.
# Ideally the CountVectorizer should have been applied to the cleaned up corpus in Task 2
my_cv = CountVectorizer(stop_words='english', ngram_range=(1,1), max_df = 5, min_df = 5)
my_data_cv = my_cv.fit_transform(data_clean2.body)
my_data_dtm = pd.DataFrame(my_data_cv.toarray(), columns=my_cv.get_feature_names())
my_data_dtm.index = data_clean2.index


data = data_dtm.transpose()
data.head()
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
'''
for movie, top_words in top_dict.items():
    print(movie)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')
'''

from collections import Counter

# Let's first pull out the top 30 words for each comedian
words = []
for movie in data.columns:
    top = [word for (word, count) in top_dict[movie]]
    for t in top:
        words.append(t)
        
add_stop_words = [word for word, count in Counter(words).most_common() if count == 7 or count == 6]
add_stop_words.append('use')
add_stop_words.append('know')
add_stop_words.append('want')
add_stop_words.append('user')
add_stop_words.append('make')
add_stop_words.append('need')
add_stop_words.append('did')
add_stop_words.append('accounts')
add_stop_words.append('today')
add_stop_words.append('account')
add_stop_words.append('went')
add_stop_words.append('lot')
add_stop_words.append('week')
add_stop_words.append('post')
add_stop_words.append('posts')
add_stop_words.append('come')
add_stop_words.append('held')
add_stop_words.append('yes')
add_stop_words.append('used')
add_stop_words.append('di')
add_stop_words.append('nbsp')
add_stop_words.append('like')
add_stop_words.append('think')
add_stop_words.append('hive')
add_stop_words.append('apr')
add_stop_words.append('date')
add_stop_words.append('daily')
add_stop_words.append('posting')
add_stop_words.append('steem')
add_stop_words.append('steemit')
add_stop_words.append('comments')
add_stop_words.append('thanks')
add_stop_words.append('you')
add_stop_words.append('thank')
add_stop_words.append('hello')
add_stop_words.append('just')
add_stop_words.append('time')
add_stop_words.append('hi')
add_stop_words.append('hehe')
add_stop_words.append('ec')
add_stop_words.append('day')
add_stop_words.append('dont')
add_stop_words.append('th')
add_stop_words.append('im')
add_stop_words.append('eb')
add_stop_words.append('way')
add_stop_words.append('really')
add_stop_words.append('year')
add_stop_words.append('people')
add_stop_words.append('got')
add_stop_words.append('start')
add_stop_words.append('took')
add_stop_words.append('look')
add_stop_words.append('la')
add_stop_words.append('number')

from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.body)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

'''
#Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

# Reset the output dimensions
plt.rcParams['figure.figsize'] = [16, 6]

for index, continent in enumerate(data.columns):
    wc.generate(data_clean2.body[continent])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(continent)
    
plt.show()



#UNIQUE WORDS
unique_list = []
for movie in data.columns:
    uniques = data[movie].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(continentname, unique_list)), columns=['continents', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')

import numpy as np

y_pos = np.arange(len(data_words))

plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.continents)
plt.title('Number of Unique Words', fontsize=20)
plt.show()

#Single Words
data_ff_words = data.transpose()[['medal', 'japan', 'tokyo', 'olympic','athlete','olympics']]
data_ff = pd.concat([data_ff_words.medal, data_ff_words.japan,data_ff_words.tokyo, data_ff_words.olympic,data_ff_words.athlete,data_ff_words.olympics], axis=1)
data_ff.columns = ['medal', 'japan', 'tokyo', 'olympic','athlete','olympics']
data_ff

'''

#sentiment analysis
data = data_df
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['body'].apply(pol)
data['subjectivity'] = data['body'].apply(sub)
plt.rcParams['figure.figsize'] = [10, 8]
for index, continent in enumerate(data.index):
    x = data.polarity.loc[continent]
    y = data.subjectivity.loc[continent]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, continent, fontsize=10)
    plt.xlim(.09, .26) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()



'''
#topic modeling
data = data_stop
from gensim import matutils, models
import scipy.sparse
tdm = data.transpose()
tdm.head()
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
import nltk
from nltk import word_tokenize, pos_tag
def nouns(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)
data_nouns = pd.DataFrame(data_clean.body.apply(nouns))
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said','hehe','thank',
                  'day','therefore','lot','yesterday','comments','post','posts','friend',
                  'level','aa','aaa','aagy','steemit','steem','today','thanks',
                  'hello','week','way','morning','blockchain','ha','greetings','number','years',
                  'users','account','place','friends','support',
                  'support','home','house','things','share','world','block',
                  'venezuela','group','picture','photo','year','days','eb','ec','appics','que',
                  'night','location','ed','ea','bc','crypto','bitcoin','version','congratulations',
                  'app','che','use','view','cc'
                  ]
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.body)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)
ldan.print_topics()

def nouns_adj(text):
  # Type your code to complete the function.
  # 1 point for applying for NN and JJ
  is_noun = lambda pos: pos[:2] == 'NN' # or the students can nouns from text using nouns(text)
  is_adj =  lambda pos: pos[:2] == 'JJ'
  tokenized = word_tokenize(text)
  all_nouns_adj = [word for (word, pos) in pos_tag(tokenized) if (is_noun(pos) or is_adj(pos))]
  # 1 point returning the correct thing
  return ' '.join(all_nouns_adj)

data_nouns_adj = pd.DataFrame(data_clean.body.apply(nouns_adj))

# Recreate a document-term matrix with only nouns/adjs
cvna = CountVectorizer(stop_words=stop_words)
data_cvna = cvna.fit_transform(data_nouns_adj.body)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index

corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

# 1 point to apply the `nouns_adj` function to the reviews and determine the 3 best topics
ldana = models.LdaModel(corpus=corpusna, num_topics=3, id2word=id2wordna, passes=10)
ldana.print_topics()
'''