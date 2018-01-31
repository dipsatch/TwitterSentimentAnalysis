import numpy as np
import pandas as pd
import os
import pickle
import sys
obama_df = pd.read_csv("data/obama_csv_test.csv", sep='\t', encoding='latin1')
obama_df = obama_df[pd.notnull(obama_df['label'])]
obama_df['label'] = obama_df['label'].astype(np.int)

texts = []  # list of text samples
labels_index = {-1: 0,
				0: 1,
				1: 2}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

obama_df = obama_df[obama_df['label'] != 2]
nb_rows = len(obama_df)
for i in range(nb_rows):
	row = obama_df.iloc[i]
	texts.append(str(row['tweet']))
	labels.append(labels_index[int(row['label'])])

texts = np.asarray(texts)
labels = np.asarray(labels)

def tokenize(texts):
    '''
    For SKLearn models / XGBoost / Ensemble, use CountVectorizer to generate
    n-gram vectorized texts efficiently.

    Args:
        texts: input text sentences list

    Returns:
        the n-gram text
    '''
    if os.path.exists('data/vectorizer.pkl'):
        with open('data/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            x_counts = vectorizer.transform(texts)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x_counts = vectorizer.fit_transform(texts)

        with open('data/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    '''
    Perform TF-IDF transform to normalize the dataset

    Args:
        x_counts: the n-gram tokenized sentences

    Returns:
        the TF-IDF transformed dataset
    '''
    if os.path.exists('data/tfidf.pkl'):
        with open('data/tfidf.pkl', 'rb') as f:
            transformer = pickle.load(f)
            x_tfidf = transformer.transform(x_counts)
    else:
        transformer = TfidfTransformer()
        x_tfidf = transformer.fit_transform(x_counts)

        with open('data/tfidf.pkl', 'wb') as f:
            pickle.dump(transformer, f)

    return x_tfidf


print(['hello and Hi hello'])
x_counts = tokenize(['hello and Hi hello'])
# print(obama_df)
# print(len(obama_df))
print(x_counts)
# sys.exit("bye")
data = tfidf(x_counts)
print(data)
print('-' * 80)