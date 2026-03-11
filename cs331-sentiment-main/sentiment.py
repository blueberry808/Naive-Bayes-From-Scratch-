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
        cleaned = re.sub(r'[^\w\s]', '', line).lower()
        tokens = cleaned.split()
        processed_lines.append(tokens)

    return processed_lines

def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    vocab_set = set()

    for line in preprocessed_text:
        for word in line[:-1]:   # exclude label only
            vocab_set.add(word)

    return sorted(list(vocab_set))



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
    return vectors, labels, vocab



def main():
    # Take in text files and outputs sentiment scores
    with open("../trainingSet.txt", "r", encoding="utf-8") as f: 
        training_data = f.read()
    with open("../testSet.txt", "r", encoding="utf-8") as f: 
        test_data = f.read()
    
    train_vectors, train_labels, train_vocab = hi(training_data, "train")
    test_vectors, test_labels, test_vocab = hi(test_data, "test")
    
    
    classifier = BayesClassifier()
    
    file_length = 499
    file_sections = [file_length // 4, file_length // 3, file_length // 2]
    
    
    #499
    #499
    #1359
    print(len(train_vectors))
    print(len(train_labels))
    print(len(train_vocab))
    
    #splitting up data into for equal parts
    #1/4
    p1_data = train_vectors[:file_sections[0]]
    p1_labels = train_labels[:file_sections[0]]
    classifier.train(p1_data, p1_labels,train_vocab) #train
    pred1 = classifier.classify_text(train_vectors, train_vocab) #test on train data
    print(f"Accuracy from training on 1/4 of the training data: {accuracy(pred1, train_labels)}")

    #half 
    p2_data = train_vectors[:file_sections[2]]
    part2_labels = train_labels[:file_sections[2]]
    classifier.train(p2_data, part2_labels,train_vocab)
    pred2 = classifier.classify_text(train_vectors, train_vocab) #test on train data
    print(f"Accuracy from training on 2/4 of the training data: {accuracy(pred2, train_labels)}")

    #three quarters
    p3_data = train_vectors[:3*file_sections[0]]
    part3_labels = train_labels[:3*file_sections[0]]
    classifier.train(p3_data, part3_labels,train_vocab)
    pred3 = classifier.classify_text(train_vectors, train_vocab) #test on train data
    print(f"Accuracy from training on 3/4 of the training data: {accuracy(pred3, train_labels)}")

    #whole data
    p4_data = train_vectors
    part4_labels = train_labels
    classifier.train(p4_data, part4_labels,train_vocab)
    pred4 = classifier.classify_text(train_vectors, train_vocab) #test on train data
    print(f"Accuracy from training on all of the training data: {accuracy(pred4, train_labels)}")

    return 1

if __name__ == "__main__":
    main()