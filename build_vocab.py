# Imports
import pandas as pd
import numpy as np
import pickle
import nltk
import string
from gensim.models import KeyedVectors
from nltk.corpus import stopwords


# Main function
def main():
    # Make sure stopword libary is downloaded    
    nltk.download('stopwords')

    # Get the stop words
    stop_words = set(stopwords.words('english'))

    # Load Google's Word2Vec model
    model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

    # Load the data provided by Kaggle
    traindata = pd.read_csv('data/train_data.csv')
    trainlabel = pd.read_csv('data/train_labels.csv')

    # Remove unused rows and merge data
    traindata = traindata[['id', 'question1', 'question2']]
    trainlabel = trainlabel[['is_duplicate']]
    data = pd.concat([traindata, trainlabel], axis=1)

    # Build vocabulary (index 0 is for unused words)
    vocabulary = ['<unk>']

    # We're going to build the tokenized set
    tokenized_data = []

    # remove punctuation from strings
    translator = str.maketrans('', '', string.punctuation)

    # Helper function to tokenize a string
    def tokenize_question(question):
        split = str(question).lower().translate(translator).split()
        tokenized = []
        for word in split:
            if word in stop_words:
                continue
                
            if word not in model.vocab:
                continue
            
            if word not in vocabulary:
                vocabulary.append(word)
            
            idx = vocabulary.index(word)
            tokenized.append(idx)
            
        return tokenized

    # Loop over dataset
    for _, x in data.iterrows():
    
        # Get the questions
        token_one = tokenize_question(x['question1'])
        token_two = tokenize_question(x['question2'])
        
        # Remember longest question
        if len(token_one) > longest:
            longest = len(token_one)
            
        if len(token_two) > longest:
            longest = len(token_two)
        
        # Format question pair
        formatted = {
            'id' : x['id'], 
            'is_duplicate' : x['is_duplicate'], 
            'question1': token_one, 
            'question2': token_two 
        }
        
        # Add to data 
        tokenized_data.append(formatted)

    # Format dataframe and store it
    df = pd.DataFrame(tokenized_data)
    df.to_csv('data/tokenized_questions.csv')

    # Store the vocabulary
    pickle.dump(vocabulary, open("data/vocab.p", "wb" ))
    

# Run main function if main file
if __name__ == "__main__":
    main()