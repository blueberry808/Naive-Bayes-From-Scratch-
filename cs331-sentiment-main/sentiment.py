# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import re 
from classifier import BayesClassifier
def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    processed_text = re.sub(r'[^\w\s]','',text)
    text_list = processed_text.split() 

    return text_list


def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    vocab_list = []
    for word in preprocessed_text: 
        if word in vocab_list: 
            continue
        else: 
            vocab_list.append(word)

    #sort alphabetically 
    vocab_list = sorted(vocab_list)
    return vocab_list


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """
    size_of_vector = len(vocab) + 1 #+1 to incorporate the label 
    vectorized_text = [0] * size_of_vector
    for i in range(len(vocab)): 
        if vocab[i] in text: 
            vectorized_text[i] = 1
        else: 
            vectorized_text[i] = 0 
    vectorized_text[size_of_vector] = text[len(text)] #assign the label 
    labels = text[len(text)] #check if this is right 
    return vectorized_text, labels


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    #we're assuming len(pred_labels) = len(t_labels_)
    correct = 0 
    for i in range(len(predicted_labels)): 
        if predicted_labels[i] == true_labels[i]: 
            correct+=1 
    
    accuracy_score = correct/len(predicted_labels)
    return accuracy_score


def main():
    # Take in text files and outputs sentiment scores
    with open("trainingSet.txt", "r", encoding="utf-8") as f: 
        training_data = f.read()

    text_list = process_text(training_data)
    vocab = build_vocab(text_list)
    vectors,labels = vectorize_text(text_list, vocab)

    test = BayesClassifier()
    test()

    
    return 1


if __name__ == "__main__":
    main()