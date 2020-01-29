---
layout: post
title:      " Sentiment Analysis and Classification of Amazon, IMDb, and Yelp Reviews"
date:       2020-01-29 20:53:55 +0000
permalink:  sentiment_analysis_and_classification_of_amazon_imdb_and_yelp_reviews
---

## Motivation for this post.
Text classification is one of the fundamental tasks in Natural Language Processing (NLP) , the field concerned with  how to process and analyze large amounts of natural language data. Text is an incredibly available data type, but it is generally difficult to extract meaningful insights from text data due to its raw unstructured form. In the case where data is  provided by customers, text data can provide direct feedback to companies that affects business decisions. As a result, businesses devote great resources to structuring, processing and analyzing this type of data. 

The text classification process involves assigning labels or categories to text according to its content. Machine learning can assist this process by classifying unseen observations based on pre-labelled examples. This is possible because machine learning algorithms learn the associations between input pieces of text and a particular label. The goal of this post is to understand how machine learning choices can affect the results of a classification problem involving multiple data sources. 

## The dataset.
The dataset for this analysis is the [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences#) from the University of California-Irvine (UCI) [Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). This dataset contains 3,000 sentences labelled with positive or negative sentiment sourced from three websites:  [Amazon](https://www.amazon.com), [IMDb](https://www.imdb.com), and [Yelp](https://www.yelp.com). For each website, there are 500 positive sentences labelled by **1** and 500 negative sentences labelled by **0**. The data for each website is stored in its own *txt* file.

## Load the data.
The first step in the supervised machine learning task  of text classification is to load the data supplied by each site. This solution was developed using Python in a Jupyter Notebook. A column *source* is added to represent which site provided the data.

```
# create dataframes
import pandas as pd

# amazon
amazon = open('./Blog4/amazon_cells_labelled.txt').read()

a_labels, a_texts = [], []
for i, line in enumerate(amazon.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        a_texts.append(content[0])
        a_labels.append(content[1])

df_a = pd.DataFrame()
df_a['label'] = a_labels
df_a['text'] = a_texts
df_a['source'] = 'amazon'

# imdb
imdb = open('./Blog4/imdb_labelled.txt').read()

i_labels, i_texts = [], []
for i, line in enumerate(imdb.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        i_texts.append(content[0])
        i_labels.append(content[1])

df_i = pd.DataFrame()
df_i['label'] = i_labels
df_i['text'] = i_texts
df_i['source'] = 'imdb'

# yelp
yelp = open('./Blog4/yelp_labelled.txt').read()

y_labels, y_texts = [], []
for i, line in enumerate(yelp.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        y_texts.append(content[0])
        y_labels.append(content[1])

df_y = pd.DataFrame()
df_y['label'] = y_labels
df_y['text'] = y_texts
df_y['source'] = 'yelp'
```

Preview the dataframes and confirm their sizes.

```
# preview 
display(df_a.head())
display(df_a.shape)
display(df_i.head())
display(df_i.shape)
display(df_y.head())
display(df_y.shape)
```

![Site DataFrames](https://github.com/monstott/Blogs/raw/master/Blog4/site_df.PNG)

As anticipated, each website dataframe consists of 1,000 sentences.

Combine the 3 site dataframes together and confirm their concatenated size.

```
# combine site dataframes
df = pd.concat([df_a, df_i, df_y], ignore_index=True)
df.label = df.label.astype(int)
df.info()

> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 3000 entries, 0 to 2999
> Data columns (total 3 columns):
> label     3000 non-null int32
> text      3000 non-null object
> source    3000 non-null object
> dtypes: int32(1), object(2)
> memory usage: 58.7+ KB
```

No errors were encountered in combining the 3 sections into a working dataset with 3,000 observations.
## Explore the contents.
The next step in the process is to learn about the text content. This will be accomplished by introducing new features that reveal information about the sentences in the dataset.

#### Introduce sentence composition features.

```
# descriptive features
import string

df['chars'] = df.text.apply(len)
df['words'] = df.text.apply(lambda x: len(x.split()))
df['avg_wlen'] = df['chars'] / df['words']
df['puncs'] = df.text.apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['uppers'] = df.text.apply(lambda x: len([word for word in x.split() if word.isupper()]))
df['titles'] = df.text.apply(lambda x: len([word for word in x.split() if word.istitle()]))
df.head()
```

![Text Features](https://github.com/monstott/Blogs/raw/master/Blog4/text_features.PNG)

Definitions for the sentence composition features:

* **chars**:  total number of characters in the sentence.
* **words**:  total number of words in the sentence.
* **avg_wlen**: average word length in the sentence.
* **puncs**: total number of punctuation marks in the sentence.
* **uppers**: total number of upper case words in the sentence.
* **titles**: total number of title case words in the sentence.

Look at statistics for each of these features.

```
# descriptive statistics for sentence features
display(df.groupby(['source', 'label']).describe().loc[:,(slice(None),['mean', 'std'])].reset_index())
display(df.groupby(['source', 'label']).describe().loc[:,(slice(None),['min', 'max'])].reset_index())
```

![Text Feature Statistics](https://github.com/monstott/Blogs/raw/master/Blog4/text_feature_stats.PNG)

**Results:**

For **Amazon** reviews:

* **characters**: Mean counts for positive (53.6) and negative (56.8) reviews are nearly equal.
* **words**: Mean counts for positive (9.9) and negative (10.6) reviews are nearly equal.
* **average word**: Mean length for positive (5.7) and negative (5.7) reviews are equal.
* **punctuation**: Mean counts for positive (1.8) and negative (2.0) reviews are nearly equal.
* **upper case**: Mean counts for positive (0.41) reviews are *less* than negative (0.56). 
The maximum count for negative (15) reviews is 5 times that of positive (3).
* **title case**: Mean counts for positive (1.3) and negative (1.2) reviews are nearly equal. 

For **IMDb** reviews:

* **characters**: Mean counts for positive (87.5) reviews are *higher* than negative (77.1). 
The maximum count for positive (479) reviews is *higher* than negative (321).
* **words**: Mean counts for positive (15.1) and negative (13.6) reviews are nearly equal.
The maximum count for positive (71) reviews is *higher* than negative (56).
* **average word**: Mean length for positive (5.8) and negative (5.8) reviews are equal.
* **punctuation**: Mean counts for positive (2.7) and negative (2.5) reviews are nearly equal.
* **upper case**: Mean counts for positive (0.40) reviews and negative (0.36) reviews are nearly equal. 
* **title case**: Mean counts for positive (1.8) and negative (1.4) reviews are nearly equal. 
The maximum count for positive (13) reviews is *lower* than negative (22).

For **Yelp** reviews:

* **characters**: Mean counts for positive (55.9) reviews are *lower* than negative (60.8). 
* **words**: Mean counts for positive (10.3) and negative (11.5) reviews are nearly equal.
* **average word**: Mean length for positive (5.6) and negative (5.4) reviews are nearly equal.
* **punctuation**: Mean counts for positive (1.9) and negative (2.0) reviews are nearly equal.
The maximum count for positive (19) reviews is *higher* than negative (11).
* **upper case**: Mean counts for positive (0.30) reviews are *lower* than negative (0.50). 
The maximum count for positive (3) reviews is *lower* than negative (13).
* **title case**: Mean counts for positive (1.3) and negative (1.3) reviews are equal. 

Takeaways:

* A negative **Amazon** review is a little more likely to use upper case words than a positive review.
* A positive **IMDb** review is more likely to have a high character count than a negative review.
* A negative **Yelp** review is more likely to have a high character count and to use upper case words than a positive review..
* In general, **IMDb** reviews use more words and characters, regardless of sentiment.

#### Add part of speech features.

```
# part of speech features
import textblob
# package requirement: python -m textblob.download_corpora

pos_group = {
    'noun' : ['NN','NNS','NNP','NNPS'], # Singular Noun, Plural Noun, Proper Singular Noun, Proper Plural Noun
    'pron' : ['PRP','PRP$','WP','WP$'], # Personal Pronoun, Possessive Pronoun, Personal wh-Pronoun, Possessive wh-Pronoun
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'], # Base Verb, Past Tense, Gerund, Past Participle, 1st or 2nd, 3rd
    'adj' :  ['JJ','JJR','JJS'], # Adjective, Comparative, Superlative
    'adv' : ['RB','RBR','RBS','WRB'] # Adverb, Comparative, Superlative, wh-Adverb
}

# check POS and obtain tag count for input
def pos_counts(x, part):
    count = 0
    try:
        wiki = textblob.TextBlob(x)
        for item in wiki.tags:
            pos = list(item)[1]
            if pos in pos_group[part]:
                count += 1
    except:
        pass
    return count

# count tags in dataset
df['nouns'] = df.text.apply(lambda x: pos_counts(x, 'noun'))
df['verbs'] = df.text.apply(lambda x: pos_counts(x, 'verb'))
df['adjs'] = df.text.apply(lambda x: pos_counts(x, 'adj'))
df['advs'] = df.text.apply(lambda x: pos_counts(x, 'adv'))
df['prons'] = df.text.apply(lambda x: pos_counts(x, 'pron'))
df.head()
```

![POS Features](https://github.com/monstott/Blogs/raw/master/Blog4/pos_features.PNG)

Definitions for the part of speech features:

* **nouns**:  total number of nouns in the sentence.
* **verbs**:  total number of verbs in the sentence.
* **adjs**: total number of adjectives in the sentence.
* **advs**: total number of adverbs in the sentence.
* **prons**: total number of pronouns in the sentence.

Part of speech (POS) tagging can help to further identify differences between websites, and between positive and negative reviews. This process works by taking each sentence in the dataset and comparing it to a set of part-of-speech tags (e.g., `['NN','NNS','NNP','NNPS']`) that represents a sentence component (e.g., noun). For every word in the sentence that matches with the set of tags, the `pos_counts()` function increments a part-of-speech variable. Information on the tags used to form the parts of speech can be found [here](https://www.clips.uantwerpen.be/pages/mbsp-tags).

Look at statistics for each of the POS features.

```
# descriptive statistics for POS features
display(df.groupby(['source', 'label']).describe().loc[:,(slice('nouns','prons'),['mean', 'std'])].reset_index())
display(df.groupby(['source', 'label']).describe().loc[:,(slice('nouns','prons'),['min', 'max'])].reset_index())
```

![POS Features](https://github.com/monstott/Blogs/raw/master/Blog4/pos_feature_stats.PNG)

**Results:**

For **Amazon** reviews:

* **nouns**: Mean counts for positive (2.5) and negative (2.5) reviews are equal.
* **verbs**: Mean counts for positive (1.8) reviews are *less* than negative (2.1). 
* **adjectives**: Mean counts for positive (1.2) reviews are *greater* than negative (0.8).
* **adverbs**: Mean counts for positive (0.85) reviews are *less* than negative (1.1).
* **pronouns**: Mean counts for positive (0.96) and negative (1.1) reviews are nearly equal. 

For **IMDb** reviews:

* **nouns**: Mean counts for positive (3.9) reviews are *greater* than negative (3.1).
* **verbs**: Mean counts for positive (2.5) and negative (2.5) reviews are equal. 
* **adjectives**: Mean counts for positive (1.6) and negative (1.4) reviews are nearly equal.
* **adverbs**: Mean counts for positive (1.1) reviews are *less* than negative (1.4).
* **pronouns**: Mean counts for positive (1.2) and negative (1.0) reviews are nearly equal. 

For **Yelp** reviews:

* **nouns**: Mean counts for positive (2.6) and negative (2.5) reviews are nearly equal.
* **verbs**: Mean counts for positive (1.8) reviews are *less* than negative (2.4). 
* **adjectives**: Mean counts for positive (1.3) reviews are *greater* than negative (0.96).
* **adverbs**: Mean counts for positive (0.92) reviews are *less* than negative (1.4).
* **pronouns**: Mean counts for positive (0.89) reviews are *less* than negative (1.2). 

Takeaways:

* The biggest part-of-speech difference for **Amazon** reviews is that positive sentiments use more adjectives. 
* The biggest part-of-speech difference for **IMDb** reviews is that positive sentiments use more nouns. 
* The biggest part-of-speech difference for **Yelp** reviews is that negative sentiments use more verbs. 
* In general, **IMDb** reviews have more counts for the parts of speech.

## Clean the text.
Properly processing text is important for reducing noise present in text data. Cleaning text data increases the accuracy of results by normalizing the case and format of sentence words. It also reduces the computational load placed on models by removing unnecessary and uninformative words from the sentences.

```
# text cleaning
from nltk.tokenize import word_tokenize
import numpy as np
import string

i = 0
df['clean_text'] = ''
for row in df.text:
    # add spaces to prevent word merging
    row = row.replace('.', '. ', row.count('.')).replace(',', ', ', row.count(','))
    # split into words
    tokens = word_tokenize(row)
    # convert to lower case
    tokens = [token.lower() for token in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [token.translate(table) for token in tokens]
    # remove non-alphabetic or numeric tokens
    words = [word for word in words if word.isalnum()]
    # filter stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    #print(words)
    df['clean_text'][i] = ' '.join(words)
    i += 1
df.clean_text = df.source + ' ' + df.clean_text
df.head()
```

![Cleaned Text DataFrame](https://github.com/monstott/Blogs/raw/master/Blog4/cleaned_df.PNG)

Text cleaning procedure:

1. Add spaces after periods and commas in each sentence (to prevent word merging).
2. Split sentences into tokens separated by whitespace.
3. Convert words to lowercase. 
4. Remove punctuation tokens (```!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~```).
5. Remove non-alphabetic and non-numeric tokens.
6. Remove stop words from the tokens.
7. Combine remaining tokens back into sentences.
8. Prepend the source website onto sentences (for identification).

Text cleaning is dependent on the context of the problem. Lemmatisation is a common text cleaning step that groups together the forms of a word so they can be analysed as a single term. Although it is common, I chose not to implement that step here in case there is a relationship between word inflection and sentiment in the reviews.

Stop words provide little information content. These words do not help a model determine the meaning of an input and can be removed without causing negative consequences. The stop words removed are: 

`['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]`

## Split into training and testing sets.
The text data has been cleaned. Now I can split the dataset into a training set for the models to learn on and a testing set for making predictions and calculating performance.

```
# split dataset
from sklearn import model_selection
from sklearn import preprocessing

# train-test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(df.clean_text, df.label) 

# label encode the target 
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
```

## Create vector matrices.
Training a machine learning classifier requires the transformation of each text into a numerical representation.  I'll construct a few versions with different properties so that I can see how they compare in machine learning models. 

#### Count vector.
Count Vector converts a collection of text documents into a matrix of token counts.  The output is a matrix in which every row is a document (sentences in this case), every column is a term, and every cell is the frequency count of a term in a document.

```
# count vector
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}') 
count_vect.fit(df.clean_text) # regexp selects tokens of 1 or more alphanumeric characters

xall_count = count_vect.transform(df.clean_text)
xtrain_count = count_vect.transform(x_train)
xtest_count = count_vect.transform(x_test)
```

#### TF-IDF vectors.
The Term Frequency-Inverse Document Frequency (TF-IDF) vectors calculate how important the terms are for each document (sentences in this case). The calculation is based on how many times a term appears in the document and how many other documents have that same term. High TF-IDF scores indicate that a term is important to a document.
The TF-IDF formula is made of two components:

* **Term Frequency - TF** = [*# of Specific Term*] / [*# of All Terms*] 
* **Inverse Document Frequency - IDF** = log( [*# of Total Documents*] / [*# of Documents with SpecificTerm*] )

These components are then multiplied together in the formula: **TF-IDF** = **TF · IDF**

TF-IDF vectors can be generated for characters, words, or n-grams. N-grams are combinations of adjacent words in a given text, where n is the number of words incuded in each token. In the `TfidfVectorizer()` function, an n-gram range parameter is used to set the lower and upper bounds of the n-grams. I'll calculate TF-IDF vectors for each word and for n-grams with a lower bound of 2 words and an upper bound of 3 words.

```
# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# word-level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df.clean_text)
xtrain_tfidf = tfidf_vect.transform(x_train)
xtest_tfidf = tfidf_vect.transform(x_test)

# ngram-level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram.fit(df.clean_text) # measures bi-grams and tri-grams
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(x_train)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(x_test)
```
## Identify text topics.
Topic modelling is an unsupervised matrix factorization technique that is used to identify groups of words (that together make a topic) within a collection of texts. Topics are repeating patterns of co-occurring terms in a collection. Latent Dirichlet Allocation (LDA) is an iterative model that starts with a fixed number of topics where topics are represented as a probability distribution over the terms. In turn, each document is represented as a probability distribution over topics. Together, these distributions provide a sense of the ideas within the documents. The goal of LDA is to figure out which topics create the input documents.

LDA works by converting a document-term frequency matrix into two lower-dimensional matrices: 

* a document-topics matrix (number of documents by number of topics) and
* a topic-terms matrix (number of topics by number of terms)

These two matrices are then optimized through iterative sampling. In every iteration, LDA attempts to adjust the topic assignment for each term in each document. For all terms and topics, a topic-term probability (**P**) is calculated from the product of two terms:

* the proportion of terms in the document that are currently assigned to that topic (**P1**) and
* the proportion of assignments to that topic in all documents that come from that term (**P2**)

These values represent the probability that a particular topic generates a specific term. If the probability (**P** = **P1· P2**)  of a new topic is higher for a specific term then the topic assignment for that term is updated. After enough iterations, document-topic and topic-term probability distributions will stabilize, signaliing completion of the LDA algorithm.

I'll construct a model that gives the top 10 terms for 10 topics.

```
# Latent Dirichlet Allocation model (with online variational Bayes algorithm)
from sklearn import decomposition

lda_model = decomposition.LatentDirichletAllocation(n_components=10, learning_method='online', max_iter=100)
lda_fit = lda_model.fit_transform(xall_count)
topics = lda_model.components_ 
vocab = count_vect.get_feature_names()

# top keywords for each topic
n_words = 10
vocab = count_vect.get_feature_names()
keywords = np.array(vocab)
topic_keywords = []
for topic_weights in topics:
    top_keyword_locs = (-topic_weights).argsort()[:n_words]
    topic_keywords.append(keywords.take(top_keyword_locs))
df_topic_kw = pd.DataFrame(topic_keywords)
df_topic_kw.columns = ['Word '+str(i) for i in range(df_topic_kw.shape[1])]
df_topic_kw.index = ['Topic '+str(i) for i in range(df_topic_kw.shape[0])]
df_topic_kw
```

![10 Words for 10 Topics](https://github.com/monstott/Blogs/raw/master/Blog4/topic_words.PNG)

Find the dominant topic for each document.

```
# dominant topic for each matrix
topic_names = ['Topic ' + str(i) for i in range(lda_model.n_components)]
df_doctop = pd.DataFrame(np.round(lda_fit, 2), columns=topic_names, index=df.index)
dominant_topic = np.argmax(df_doctop.values, axis=1)
df_doctop['dominant_topic'] = dominant_topic 
df_doctop['source'] = df.source
df_doctop.head(10)
```

![Dominant Topic](https://github.com/monstott/Blogs/raw/master/Blog4/dominant_topics.PNG)

Plot the dominant topic by source and sentiment.

```
# plot the dominant topic
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

df_doctop.groupby(['dominant_topic', 'source'])['source'].count().unstack().\
    plot(kind='bar', figsize=(15, 8), fontsize=14, ax=ax, cmap=plt.cm.get_cmap('Accent'))
ax.set_title('Document Dominant Topics by Source', fontsize=18)
ax.set_xlabel('Dominant Topic', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
ax.legend(fontsize=14);
```

![Dominant Plot](https://github.com/monstott/Blogs/raw/master/Blog4/dominant_plots.png)

**Results:**

For **Amazon** reviews:
* The dominant topic is **Topic 5**. 
* Both negative and positive sentiments have over 350 sentences (70%) with this topic. 
* Relevant top **Topic 5** keywords are are `[amazon, phone, like, great, service, really, good, bad]`.

For **IMDb** reviews:
* The dominant topics are **Topic 0** and **Topic 3**. 
* There are around 100 negative (20%) and 150 positive (30%) **Topic 0** sentences. 
* Relevant top **Topic 0** keywords are `[imdb, film, go, movie, characters, going, made, way, movies, two]`.
* There are around 150 negative (30%) and 100 positive (20%) Topic 3 sentences. 
* Relevant top **Topic 3** keywords are `[imdb,	movie, acting, plot, character, little, special, enjoyed, worse, man]`.

For **Yelp** reviews:
* The dominant topics are **Topic 5**, **Topic 6**, and **Topic 7**.
* There are around 150 negative (30%) and 100 positive (20%) **Topic 5** sentences. 
* Relevant top **Topic 5** keywords are `[yelp, phone, like, great, service, really, good, bad]`.
* There are around 50 negative (10%) and 75 positive (15%) **Topic 6** sentences. 
* Relevant top **Topic 6** keywords are `[yelp, food, amazing, love, friendly, lot, staff, great,	job, day]`.
* There are around 125 negative (25%) and 175 positive (35%) **Topic 7** sentences. 
* Relevant top **Topic 7** keywords are `[yelp,	place, good, better, sound, food, excellent, great]`.

Takeaways:

* The LDA model does an excellent job of constructing topics that split the text data by source website. 
* This is especially evident in the case of **Amazon** review sentiments for Topic 5. This is the most dominant of the 10 topics for over 70% of **Amazon** sentences. 
* Good LDA source separation is also seen in Topic 0 and Topic 3 for **IMDb**.  The keywords in these topics are related to movies and their viewing. 
* There is overlap in Topic 5 for **Amazon** and **Yelp**. The top keywords for this topic appear to concern customer service quality. Given the subjects of **Amazon** and **Yelp** reviews (products and restaurants), the overlap for this topic is expected. 
* The LDA does not split topics by sentiment well, however. Dominant positive and negative sentiment topics are identical for each source, although their percentages differ moderately.

##  Compare keywords with word embeddings.
Word embeddings models are used to predict the context surrounding a desired word. Word embeddings are a dense vectorization strategy that computes word vectors from a collection of text data by training a simple neural network. This neural network results in a high-dimensional embedding space where each term in a collection has a unique vector and the position of that vector relative to other vectors captures semantic meaning.  The position of a word within the vector space is learned from the text and is based on the surrounding words where it is used. 

The word embedding Word2Vec neural network is composed of an input layer, a hidden layer, and an output layer using the softmax activation function for multiclass classification. The input layer of the network contains one neuron for every unique word in the vocabulary. The hidden layer neurons have a unique weight for each of the input vocabulary words and a neuron count set by the desired word vector size. Once the model is trained, the hidden layer serves as a lookup table that provides the word vector for an input word as the output.

I'll train a Word2Vec word embedding on the cleaned text data (instead of using pre-trained vectors) and compare the result with the LDA model.

```
# collapse texts into a set
data = df.clean_text.map(word_tokenize).values
total_vocabulary = set(word for line in data for word in line)
print('Unique tokens in texts: {}'.format(len(total_vocabulary)))

> Unique tokens in texts: 5120
```

There are 5,120 unique tokens that will be used to build the word embeddings. The word vectors size is set to 100 with a context window that will focus on 5 words at a time. 

```
# word embedding
from gensim.models import Word2Vec

w2v_model = Word2Vec(data, size=100, window=5, min_count=1, workers=4)
w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=10)
word_vectors = w2v_model.wv
```

Words most similar to **Amazon**:

```
# amazon similar words
print('Words Most Similar to Amazon:')
display(word_vectors.most_similar('amazon'))

> Words Most Similar to Amazon:
> [('well', 0.99990),
>  ('bought', 0.99989),
>  ('phone', 0.99989),
>  ('use', 0.99989),
>  ('new', 0.99989),
>  ('case', 0.99988),
>  ('works', 0.99988),
>  ('since', 0.99988),
>  ('without', 0.99988),
>  ('right', 0.99987)]
```

Words most similar to **IMDb**:

```
# imdb similar words
print('Words Most Similar to IMDB:')
display(word_vectors.most_similar('imdb'))

> Words Most Similar to IMDb:
> [('film', 0.99995),
> ('made', 0.99995),
>  ('really', 0.99995),
>  ('even', 0.99995),
>  ('one', 0.99995),
>  ('also', 0.99994),
>  ('see', 0.99994),
>  ('still', 0.99994),
>  ('enough', 0.99994),
>  ('people', 0.99993)]
```
 
 Words most similar to **Yelp**:
 
```
print('Words Most Similar to Yelp:')
display(word_vectors.most_similar('yelp'))

> Words Most Similar to Yelp:
> [('never', 0.99993),
> ('say', 0.99992),
> ('amazing', 0.99992),
> ('going', 0.99991),
> ('character', 0.99991),
> ('little', 0.99991),
> ('place', 0.99991),
> ('film', 0.99991),
> ('nothing', 0.99991),
> ('many', 0.99990)]
```

Similarity between sources:

```
print('Cosine Similarity between Amazon and IMDB:')
display(word_vectors.similarity('amazon', 'imdb'))

print('Cosine Similarity between Amazon and Yelp:')
display(word_vectors.similarity('amazon', 'yelp'))

print('Cosine Similarity between IMDB and Yelp:')
display(word_vectors.similarity('imdb', 'yelp'))

> Cosine Similarity between Amazon and IMDB:
> 0.99983
> Cosine Similarity between Amazon and Yelp:
> 0.99976
> Cosine Similarity between IMDB and Yelp:
> 0.99989
```

**Results:**

* For **Amazon**, a word shared between the LDA dominant topic (Topic 7) and word embeddings most similar words is `[phone]`. 
* For **IMDb**, words shared between the LDA dominant topics (Topic 0, Topic 3) and word embeddings most similar words are `[film, made]`. 
* For **Yelp**, words shared between the LDA dominant topics (Topic 5, Topic 6, Topic 7) and word embeddings most similar words are `[amazing, place]`. 
* The cosine similarity score, which measures the cosine of the angle between word vectors, is very high between sources. This helps to justify the overlap in word embeddings most similar words and LDA topic keywords. 

## Fit models.
Now that the dataset has been processed and explored, I will create classifiers using some of the features that have been engineered. Many machine learning models will be trained so that their performance can be compared. A wrapper function will be used to streamline the model creation process.

```
# model wrapper function
from sklearn import metrics

def train_model(classifier, train_features, label, test_features):
    # fit the training data on classifier
    classifier.fit(train_features, label)
    
    # predict testing data labels
    predictions = classifier.predict(test_features)
    
    return metrics.accuracy_score(predictions, y_test)
```

#### Naive Bayes.
Naive Bayes (NB) is a classification technique that uses a supervised machine learning algorithm based on Bayes’ Theorem.  An NB classifier assumes that the presence of class features are independent and unrelated. Bayes' Theorem calculates the posterior probability of a class given a predictor **P(c | x)** from 3 terms: the prior probability of that class **P(c)**; the prior probability of that predictor **P(x)**;  and,  the probability of a predictor given a class (likelihood) **P(x | c)**: 

* **P(c | x)** = **( P(x | c) · P(c) ) / P(x)**. 

The posterior probability is calculated for each class and the class with the highest outcome becomes the prediction. I'll use this classifier on each of the 3 count vectorization forms created.

```
# Naive Bayes
from sklearn import naive_bayes

# Count Vectors
nb_cv = train_model(naive_bayes.MultinomialNB(), xtrain_count, y_train, xtest_count)
print("[Naive Bayes] Count Vectors Accuracy:", round(nb_cv, 3))

# Word-Level TF-IDF Vectors
nb_wl = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, y_train, xtest_tfidf)
print("[Naive Bayes] Word-Level TF-IDF Accuracy:", round(nb_wl, 3))

# Ngram-Level TF-IDF Vectors
nb_nl = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
print("[Naive Bayes] N-Gram-Level TF-IDF Accuracy:", round(nb_nl, 3))

> [Naive Bayes] Count Vectors Accuracy: 0.819
> [Naive Bayes] Word-Level TF-IDF Accuracy: 0.809
> [Naive Bayes] N-Gram-Level TF-IDF Accuracy: 0.663
```

#### Logistic Regression.
Logistic Regression (LR) measures the relationship between a categorical dependent variable and independent variables by estimating probabilities using a logistic sigmoid function (a function with an *S*-shaped curve). The logistic function estimates the Logit function, a function which provides the logarithm of the odds in favor of an event. The value of the Logit function moves toward infinity as probability estimates approach 1. Likewise, the value of the Logit function moves toward negative infinity as the probability estimates approach 0. 

LR works by optimizing the Log Likelihood function. The Likelihood function measures the goodness of fit for data in statistical model for values of unknown parameters. The peak of the Likelihood function represents the combination of model parameter values that maximizes the probability of drawing the data. The Log Likelihood function is usually used in the optimization process due to its convenient form when trying to find the peak.

The output of LR is the assignment of a probability between 0 (negative sentiment) and 1 (positive sentiment) for every text. The probability value can vary between certainty of a negative sentiment (0) and certainty of a positive sentiment (1). This algorithm becomes a classifier when it uses a cutoff probability value (0.5). Inputs with a resulting probability greater than the cutoff become predictions of class 1 and inputs with probability below the cutoff become predictions of class 0.

```
# Logistic Regression
from sklearn import linear_model

# Count Vectors
lr_cv = train_model(linear_model.LogisticRegression(), xtrain_count, y_train, xtest_count)
print("[Logistic Regression] Count Vectors Accuracy:", round(lr_cv, 3))

# Word-Level TF-IDF Vectors
lr_wl = train_model(linear_model.LogisticRegression(), xtrain_tfidf, y_train, xtest_tfidf)
print("[Logistic Regression] Word-Level TF-IDF Accuracy:", round(lr_wl, 3))

# Ngram-Level TF-IDF Vectors
lr_nl = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
print("[Logistic Regression] N-Gram TF-IDF Accuracy:", round(lr_nl, 3))

> [Logistic Regression] Count Vectors Accuracy: 0.803
> [Logistic Regression] Word-Level TF-IDF Accuracy: 0.808
> [Logistic Regression] N-Gram TF-IDF Accuracy: 0.68
```

#### Suport Vector Machine.
Support Vector Machine (SVM) is a supervised machine learning algorithm that determines the best possible hyperplane that separates two classes. SVM works by plotting each data point in n-dimensional space, where n is the number of features. Classification results from the hyperplane separation between the classes of input data. The best hyperplane maximizes the distance between the nearest data points (support vectors) in either class. 

The distance between the nearest points of either class and the hyperplane is called the margin. The higher the margin, the more robust the model and the lower the chance of misclassification. SVM classifies classes as accurately as possible before maximizing the margin. This behavior can lead to misclassification or the need to use different kernels in cases where class separation is non-linear. For non-linear class separations, a kernel function transforms the low-dimensional input space to a  higher-dimensional space in order to make the problem separable.

```
# Support Vector Machines
from sklearn import svm

# Count Vectors
svm_cv = train_model(svm.SVC(), xtrain_count, y_train, xtest_count)
print("[Support Vector Machines] Count Vectors Accuracy:", round(svm_cv, 3))

# Word-Level TF-IDF Vectors
svm_wl = train_model(svm.SVC(), xtrain_tfidf, y_train, xtest_tfidf)
print("[Support Vector Machines] Word-Level TF-IDF Accuracy:", round(svm_wl, 3))

# Ngram-Level TF-IDF Vectors
svm_nl = train_model(svm.SVC(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
print("[Support Vector Machines] N-Gram TF-IDF Accuracy:", round(svm_nl, 3))

> [Support Vector Machines] Count Vectors Accuracy: 0.493
> [Support Vector Machines] Word-Level TF-IDF Accuracy: 0.493
> [Support Vector Machines] N-Gram TF-IDF Accuracy: 0.493
> 
```

#### Random Forest.
Random Forest (RF) models are a type of ensemble learning method for classification that constructs a multitude of decision trees and outputs class predictions from the class mode of the individual tree results.
In doing so, RF correct for overfitting in decision trees and reduces variance.

 Decision trees have a tree-like structure in which each internal node is a predictor event, each branch is the event outcome, and each leaf node is a class label. The path from root to leaf defines the rules used in making a classification determination. RF use the bagging ensemble algorithm to repeatedly select a random sample of both the training set and the features (with replacement) as inputs for each decision tree in the ensemble.

```
# Random Forest
from sklearn import ensemble

# Count Vectors
rf_cv = train_model(ensemble.RandomForestClassifier(), xtrain_count, y_train, xtest_count)
print("[Random Forest] Count Vectors Accuracy:", round(rf_cv, 3))

# Word-Level TF-IDF Vectors
rf_wl = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, y_train, xtest_tfidf)
print("[Random Forest] Word-Level TF-IDF Accuracy:", round(rf_wl, 3))

# Ngram-Level TF-IDF Vectors
rf_nl = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
print("[Random Forest] N-Gram TF-IDF Accuracy:", round(rf_nl, 3))

> [Random Forest] Count Vectors Accuracy: 0.779
> [Random Forest] Word-Level TF-IDF Accuracy: 0.749
> [Random Forest] N-Gram TF-IDF Accuracy: 0.645
```

#### Extreme Gradient Boosting.
Extreme Gradient Boosting (XGB) is a type of ensemble model that builds predictions from a group of weak prediction models (typically decision trees) that are iteratively converted from weak learners into strong learners. A weak learner is a classifier with results that have a low correlation with the true labels. Compared to other gradient boosting algirhtms, XGB focuses on computational speed and model performance. It can reduce bias and variance, but is also susceptible to overfitting.

Where bagging takes a random sample of data to use within learning algorithms, boosting instead selects sample with an additional layer of scrutiny. Boosting is an ensemble technique where new models correct for errors made in existing models by using a gradient descent algorithm to minimize a loss function in new models. Subsequent models are added until no improvements can be made. The classifications within the final model are the predictions made by XGB.

```
# Extreme Gradient Boosting
import xgboost

# Count Vectors
xgb_cv = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), y_train, xtest_count.tocsc())
print("[Xtreme Gradient Boosting] Count Vectors Accuracy:", round(xgb_cv, 3))

# Word-Level TF-IDF Vectors
xgb_wl = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), y_train, xtest_tfidf.tocsc())
print("[Xtreme Gradient Boosting] Word-Level TF-IDF: ", round(xgb_wl, 3))

# Ngram-Level TF-IDF Vectors
xgb_nl = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
print("[Xtreme Gradient Boosting] N-Gram TF-IDF Accuracy:", round(xgb_nl, 3))

> [Xtreme Gradient Boosting] Count Vectors Accuracy: 0.721
> [Xtreme Gradient Boosting] Word-Level TF-IDF:  0.711
> [Xtreme Gradient Boosting] N-Gram TF-IDF Accuracy: 0.58
```

## Compare models.
Collect model scores for the percentage of sentiments in the testing set that were predicted with the correct label (accuracy).

```
# model performance table
pd.DataFrame([[nb_cv, nb_wl, nb_nl],
              [lr_cv, lr_wl, lr_nl],
              [svm_cv, svm_wl, svm_nl],
              [rf_cv, rf_wl, rf_nl],
              [xgb_cv, xgb_wl, xgb_nl]], 
columns=['Count Vector', 'Word TF-IDF', 'n-Gram TF-IDF'], 
index=['Naive Bayes', 'Logistic Regression', 'Support Vector Machines', 'Random Forest', 'Xtreme Gradient Boosting']).round(3)
```

![Model Accuracy](https://github.com/monstott/Blogs/raw/master/Blog4/models_accuracy.PNG)

**Conclusions:**

* The model with the highest accuracy score in predicting the sentiment of testing set data is **Naive Bayes**  *with* **Count Vector** (82%) as the feature.
* The next highest model-feature scores are for **Naive Bayes** *with* **Word TF-IDF** (81%) and **Logistic Regression** *with* **Word TF-IDF** (81%). 
* The model with the lowest score is **Support Vector Machines** (49%) with an accuracy independent of the vectorization strategy selected.
* Across models, the vectorization strategy with the highest ability to predict sentiment is a tie between **Count Vector** (72%) and **Word TF-IDF** (71%). The average accuracy for **n-Gram TF-IDF** (61%) is lower by 15 percent.
* Across vectorization strategies, the model with the highest ability to predict sentiment is a tie between **Naive Bayes** (76%) and **Logistic Regression** (76%). The ranking order for the other models is **Random Forest** (72%), **Extreme Gradient Boosting** (67%), and **Support Vector Machines** (49%). 

From these results and the ease of implementing its vectorization strategy and fitting the model, I would recommend using  the **Naive Bayes** model with **Count Vectorization** as the feature for the problem of identiying sentiments in mixed source datasets.

## Final thoughts.
Classification of text is a broad task that leads in many different directions. In this investigation, I have shown how reviews from 3 different websites can be analyzed using features related to sentence structure or by part of speech. These features were able to shed some light on the differences between positive and negative reviews, and between sources. Sentiments were also looked at through the lens of topics. First, sentences were cleaned and vectorized. Then, the LDA algorithm found out which groups of words best work together as topics explaining the documents. After this, results were compared with word embeddings to see if collapsing the documents into a single set would result in some of the same relationships being formed between words. Lastly, 3 matrix vectorization strategies (*count vector, word TF-IDF, and n-gram TF-IDF*) were compared in 5 models (*Naive Bayes, Logistic Regression, Support Vector Machine, Random Forest, and Extreme Gradient Boosting*) to determine which machine learning algorithm is most appropriate in this scenario. This activity has taught me about many machine learning options for text classification. As a result of this examination of some of the choices available, I feel better equipped to tackle classification quesitions in the future. 
