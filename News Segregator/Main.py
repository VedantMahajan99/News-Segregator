
import pandas as pd
import numpy as np
from articles import articles , actual_topics
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# preprocess articles
processed_articles = []

for items in articles:
  processed_articles.append(preprocess_text(items))

# initialize and fit CountVectorizer
# The CountVectorizer object is fit (trained) and transformed (applied) on the corpus of data, returning the term frequencies for each term-document pair
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(processed_articles)

# convert counts to tf-idf
# TfidfTransformer is up to the task of converting your bag-of-words model to tf-idf
transformer = TfidfTransformer(norm=None)
tfidf_scores_transformed = transformer.fit_transform(counts)

# DIRECTLY OBTAINING TDIF SCORES
# initialize and fit TfidfVectorizer --> calculate the tf-idf values for each term-document pair in our corpus
vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(processed_articles)


# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  print("\n\n\n-------------------------------------WORD COUNT FOR EACH ARTICLE----------------------------\n\n")
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  print("\n\n\n---------------------- TDIDF SCORES TRANSFORMED FROM BAG-OF-WORDS MODEL----------------------\n\n")
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  print("\n\n\n---------------------- DIRECTLY OBTAINING TDIDF SCORES USING TfidfVectorize----------------------\n\n")
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass


# check if tf-idf scores are equal

if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'\n\nAre the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'\n\nAre the tf-idf scores the same?':['No, something is wrong :(']}))

print("\n\n")

for i in range(1,16):
  print(f'Actual topic of the Article {i} is -> {actual_topics[i-1]} .')

print("\n\nTopics of the news articles detected using TFIDF scores :\n")
# get highest scoring tf-idf term for each article
for i in range(1,16):
  print( df_tf_idf[[f'Article {i}']].idxmax())

