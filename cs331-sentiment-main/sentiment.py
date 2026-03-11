# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import re 
from classifier import BayesClassifier
def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    lines = text.split('\n')

    processed_lines = []

    for line in lines:
        cleaned = re.sub(r'[^\w\s]', '', line)
        tokens = cleaned.split()
        processed_lines.append(tokens)

    return processed_lines

def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    vocab_list = []
    for line in preprocessed_text:
        for n in range(len(line)-1): 
            word = line[n]
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
    
    vectors = []
    
    # hasWord1, hasWord2, hasWord3, ... hasWordN, Label
    
    labels = []
    
    for line in text:
        vectorized_text = [0] * size_of_vector
        if(len(line) == 0): 
            continue
        label = int(line.pop())
        
        for i in range(len(vocab)): 
            if vocab[i] in line: 
                vectorized_text[i] = 1
            else: 
                vectorized_text[i] = 0 
                
        vectorized_text[size_of_vector-1] = label #assign the label 
        labels.append(label)
        vectors.append(vectorized_text)
        
    return vectors, labels


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

def hi(file, dataset_type):
    preprocessed_text = process_text(file)
    vocab = build_vocab(preprocessed_text)
    vectors, labels = vectorize_text(preprocessed_text, vocab)

    with open("preprocessed_"+dataset_type+".txt", "w") as f:
        f.write(",".join(vocab) + ",classlabel\n")
        for vector in vectors:
            f.write(",".join(str(n) for n in vector)+"\n")
    return labels, vocab



def main():
    # Take in text files and outputs sentiment scores
    with open("../trainingSet.txt", "r", encoding="utf-8") as f: 
        training_data = f.read()
    with open("../testSet.txt", "r", encoding="utf-8") as f: 
        test_data = f.read()
    
    train_labels, train_vocab = hi(training_data, "train")
    test_labels, test_vocab = hi(test_data, "test")
    
    
    classifier = BayesClassifier()
    classifier.train(training_data, train_labels,train_vocab)

    
    return 1


if __name__ == "__main__":
    main()