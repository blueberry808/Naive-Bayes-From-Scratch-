# This file implements a Naive Bayes Classifier
import math

class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.postive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_sentences = 0
        self.percent_negative_sentences = 0
        self.file_length = 499
        self.total_positive_sentences = 0 
        self.total_negative_sentences = 0

    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab

        """

        total_positive_sentences = 0 
        total_negative_sentences = 0 
        total_num_sentences = len(train_data) 

        #we need this to perform naive bayes calculations
        for i in train_labels: 
            if i ==1: 
                total_positive_sentences += 1
            elif i==0: 
                total_negative_sentences += 1 
        
        # print("total")
        # print(total_num_sentences)
        # print("positive")
        # print((total_positive_sentences)) 
        # print("negative")
        # print((total_negative_sentences))
        
        #initialize word counts to 0
        for word in vocab:
            self.postive_word_counts[word] = 0
            self.negative_word_counts[word] = 0

        '''
        We need to find word counts for each word in the vocab for both positive and negative sentences. 
        This will allows us to perform the conditional independence probabiltiy calculations for inference. 
        '''
        #iterate thru le vectors
        for i in range(len(train_data)): 
            #iterate thru words in the vector 
            vector = train_data[i]
            label = train_labels[i]
            
            for j in range(len(vocab)):
                #print(f"Train data: {train_data}\n")
                if vector[j] == 1 and label == 1: 
                    self.postive_word_counts[vocab[j]] += 1 
                elif vector[j] == 1 and label == 0: 
                    self.negative_word_counts[vocab[j]] += 1

        self.total_positive_sentences = total_positive_sentences
        self.total_negative_sentences = total_negative_sentences
        self.percent_negative_sentences = total_negative_sentences/total_num_sentences
        self.percent_positive_sentences = total_positive_sentences/total_num_sentences
    
    #define classify_text(self: BayesClassifier, vectors: Idk, vocab: Idk) -> something:
    def classify_text(self, vectors, vocab):
        
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """

        predictions = [] 
        
        total_pos_words = 0 
        total_neg_words = 0 

        for key in self.postive_word_counts:
            total_pos_words += self.postive_word_counts[key]
        for key in self.negative_word_counts:
            total_neg_words += self.negative_word_counts[key]
        
        for sentence in vectors: 
            runSumPOS = math.log(self.percent_positive_sentences)
            runSumNEG = math.log(self.percent_negative_sentences)

            for i in range(len(vocab)):     

                if sentence[i] == 1:
                        #*Check this with teacher
                    #uniform dirichlet priors - have to add the entire vocab size to denominator since that 
                    #accounts for all positible numerator scenarios (each word in the vocab could be present or not present in a sentence)
                    pos_prob = (self.postive_word_counts[vocab[i]] + 1) / (total_pos_words + len(vocab))
                    neg_prob = (self.negative_word_counts[vocab[i]] + 1) / (total_neg_words + len(vocab))

                    runSumPOS += math.log(pos_prob)
                    runSumNEG += math.log(neg_prob)
                    

            if runSumNEG>runSumPOS: 
                predictions.append(0)
            else: 
                predictions.append(1)
        
        return predictions