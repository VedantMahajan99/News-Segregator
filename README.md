# News-Segregator

In this project I used the concept of TF-IDF LANGUAGE MODEL to detect the topic of the new article given a snippet of the article. 

TF-IDF -> Term frequency-inverse document frequency is a numerical statistic used to indicate how important a word is to each document in a collection of documents, or a corpus.
  > term frequency : How often a word appears in a document. This is the same as bag-of-words’ word count.
  > inverse document frequency :  which is a measure of how often a word appears in the overall corpus. By penalizing the         score of words that appear throughout a corpus, tf-idf can give better insight into how important a word is to a               particular document of a corpus.
  > tf-idf score : a tf-idf score for each document, representing the relevance of that word to the particular document. A         higher tf-idf score indicates a term is more important to the corresponding document.
  
 Libraries used:
 
 Pandas : pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with             “relational” or “labeled” data both easy and intuitive.
 
 sklearn : Simple and efficient tools for predictive data analysis
 * CountVectorizer : returns the term frequencies for each term-document pair
 * TfidfTransformer : task of converting your bag-of-words model to tf-idf
 * TfidfVectorizer : calculate the tf-idf values for each term-document pair in our corpus
 
 NLTK (Natural Language tool kit) : Used to process human language data. Following processes use this library
 * Text preprocessing is all about cleaning and prepping text data so that it’s ready for other NLP tasks.
 * Noise removal is a text preprocessing step concerned with removing unnecessary formatting from our text.
 * Tokenization is a text preprocessing step devoted to breaking up text into smaller units (usually words or discrete            terms).
 * Normalization is the name we give most other text preprocessing tasks, including stemming, lemmatization, upper and           lowercasing, and stopword removal.
 * Stemming is the normalization preprocessing task focused on removing word affixes.
 * Lemmatization is the normalization preprocessing task that more carefully brings words down to their root forms.
 
 Numpy : Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object,             and tools for working with these arrays.
 
 
 
  
