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
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]
        self.total_positive_sentences =0 
        self.total_negative_sentences = 0

    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab

        """

        #splitting up data into for equal parts
        #1/4
        p1_data = train_data[:self.file_sections[0]]
        p1_labels = train_labels[:self.file_sections[0]]

        #half 
        p2_data = train_data[:self.file_sections[2]]
        part2_labels = train_labels[:self.file_sections[2]]

        #three quarters
        p3_data = train_data[:3*self.file_sections[0]]
        part3_labels = train_labels[:3*self.file_sections[0]]

        #whole data
        p4_data = train_data
        part4_labels = train_labels

        training_set = [(p1_data,p1_labels), (p2_data,part2_labels), (p3_data,part3_labels), (p4_data,part4_labels)]

        for phase in range(len(training_set)): 
            data = training_set[phase][0]
            labels = training_set[phase][1]


            total_positive_sentences = 0 
            total_negative_sentences = 0 
            total_num_sentences = len(data) 

            #we need this to perform naive bayes calculations
            for i in labels: 
                if i ==1: 
                    total_positive_sentences +=1
                elif i==0: 
                    total_negative_sentences+=1 


            #We need to find word counts for each word in the vocab for both positive and negative sentences. This will allows us 
            #to perform the conditional independence probabiltiy calculations for inference. 

            # Data ["No", ]
            # Vocab 
            for i in range(len(data)): 
                for j in range(len(vocab)):
                    if data[i][j] == 1 and labels[i] == 1: 
                        self.postive_word_counts[vocab[j]] +=1 
                    elif data[i][j] == 1 and labels[i] == 0: 
                        self.negative_word_counts[vocab[j]] +=1

            self.total_positive_sentences = total_positive_sentences
            self.total_negative_sentences = total_negative_sentences
            self.percent_negative_sentences = total_negative_sentences/total_num_sentences
            self.percent_positive_sentences = total_positive_sentences/total_num_sentences
           
        predictions = self.classify_text(train_data, vocab)

        return predictions
    

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
            runSumPOS += math.log(self.percent_positive_sentences)
            runSumNEG += math.log(self.percent_negative_sentences)

            for i in range(len(vocab)):     

                if sentence[i] == 1:
                        #*Check this with teacher
                    #uniform dirichlet priors - have to add the entire vocab size to denominator since that 
                    #accounts for all positible numerator scenarios (each word in the vocab could be present or not present in a sentence)
                    runSumPOS += math.log(self.postive_word_counts[vocab[i]] +1/self.total_positive_sentences + len(vocab))
                    runSumNEG += math.log(self.negative_word_counts[vocab[i]]/self.total_negative_sentences + len(vocab))

            if runSumNEG>runSumPOS: 
                predictions.append(0)
            elif runSumPOS> runSumNEG: 
                predictions.append(1)
        
        return predictions