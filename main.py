import random
import copy
from nltk.corpus import stopwords
from random import shuffle
import pickle

# Stich the words in the n_gram together after zip() opperation
def stich_ngrams(n_grams, n):
    grams = []
    for g in n_grams:
        s = ""
        for w in g:
            s = s + w + "_"
        grams.append(s[0:len(s)-1])
    return grams

# Build array n-grams
def find_ngrams(input_list, n):
    input_list.insert(0,"<s>")
    input_list.insert(len(input_list),"<s>")
    n_grams = zip(*[input_list[i:] for i in range(n)])
    return stich_ngrams(n_grams, n)

# Utility Functions
def random_val():
    val = random.random()
    neg = random.random()
    if neg < 0.50:
        return (val * -1.0)
    else:
        return val

# Primary Functions ============================================================
def build_vocab(fd, t, remove_stop_words=False):
    index = 0
    vocab = {}
    for line in fd:
        words = line.split()

        if remove_stop_words:
            words_without = list()
            words_without.append(words[0])  # Add the label to the list
            stop = stopwords.words('english')
            for word in words[1:]:
                # Check if word is stop word
                try:
                    if word.lower() not in stop:
                        words_without.append(word)
                except UnicodeWarning:
                    print word
                    words_without.append(word)
            words = copy.deepcopy(words_without)

        if t is True:
            # Add bi-grams
            bi_grams = find_ngrams(words[1:], 2)
            for gram in bi_grams:
                lower_case_word = gram.lower()
                if not vocab.has_key(lower_case_word):
                    vocab[lower_case_word] = index
                    index = index + 1

        # Add uni-grams
        for word in words[1:]:
            lower_case_word = word.lower()
            if not vocab.has_key(lower_case_word):
                vocab[lower_case_word] = index
                index = index + 1
    print index
    return vocab, index

def build_vector_models(fd, vocab, t, remove_stop_words=False):
    vectors = []
    for line in fd:
        words = line.split()
        dic = {}
        tag = int(words[0])

        if remove_stop_words:
            words_without = list()
            words_without.append(words[0])  # Add the label to the list
            stop = stopwords.words('english')
            for word in words[1:]:
                # Check if word is stop word
                try:
                    if word.lower() not in stop:
                        words_without.append(word)
                except UnicodeWarning:
                    words_without.append(word)
            words = copy.deepcopy(words_without)

        if t is True:
            # Add bi-grams
            bi_grams = find_ngrams(words[1:], 2)
            for gram in bi_grams:
                lower_case_word = gram.lower()
                if vocab.has_key(lower_case_word):
                    dic[lower_case_word] = vocab[lower_case_word]

        # Add uni-grams
        for word in words[1:]:
            lower_case_word = word.lower()
            if vocab.has_key(lower_case_word):
                dic[lower_case_word] = vocab[lower_case_word]

        # Add dictionary and tag to the vectors list
        vectors.append((tag, dic))
    return vectors

def build_initial_weights(size):
    weights = []
    for w in range(5):
        weight = [random_val() for i in range(size)]
        weights.append(weight)
    return weights

def calculate_outcome_scores(X, bias, weights):
    predictions = [0 for i in range(5)]
    for key, value in X.iteritems():
        for i, W in enumerate(weights):
            predictions[i] = predictions[i] + W[value]
    for i, b in enumerate(bias):
        predictions[i] = predictions[i] + bias[i]
    return predictions

def update_y_hat(W, ETA, X):
    for key, value in X.iteritems():
        W[value] = W[value] - ETA
    return W

def update_y(W, ETA, X):
    for key, value in X.iteritems():
        W[value] = W[value] + ETA
    return W

def test_with_dev_data(weights, bias, vectors_dev):
    # keeps track of number correct
    count_correct = 0

    for i, (y, X) in enumerate(vectors_dev):
        # Calculate scores
        predictions = calculate_outcome_scores(X, bias, weights)
        # Determine y_hat
        y_hat = (-10,0)
        for m, p in enumerate(predictions):
            if y_hat[0] < p:
                y_hat = (p, m)
        # Add one to count if y and y_hat are the same!
        if y_hat[1] == y:
            count_correct = count_correct + 1

    return float(count_correct)/float(len(vectors_dev))

def check_convergence(previous):
    if len(previous) > 3:
        one = previous.pop()
        two = previous.pop()
        three = previous.pop()

        if one == two and two == three:
            return True
        else:
            return False

class Perceptron(object):
    def __init__(self, train_data, dev_data, use_bi_grams=False, ETA=0.05, n=1, batch=True, is_eval=False, eval_data=None):
        # Set up the perceptron algorithm
        self.batch = batch
        self.ETA = ETA
        vocab_fd = open(train_data, 'r')
        vector_fd = open(train_data, 'r')
        dev_vector_fd = open(dev_data, 'r')
        self.vocab, self.size = build_vocab(vocab_fd, use_bi_grams)
        self.vectors = build_vector_models(vector_fd, self.vocab, use_bi_grams)
        self.weights = build_initial_weights(self.size)
        # Create batch vectors to collate the updates when batch flag is True
        if batch:
            # Deep copy of the weight vectors
            self.batch_weights = copy.deepcopy(self.weights)
        # Create the bias measure
        self.bias = [1 for i in range(5)]
        # Build vector models
        self.vectors_dev = build_vector_models(dev_vector_fd, self.vocab, use_bi_grams)

        # Run the algorithum
        for i in range(0, n):
            self.run(self.weights, self.bias)
            if is_eval:
                pickle.dump((self.weights, self.bias), open( "./data/weights.p", "wb" ) )
                self.evaluate_weights(eval_data, use_bi_grams)
            # Reset weights and bias for the next run
            self.weights = build_initial_weights(self.size)
            self.bias = [1 for i in range(5)]
            # If we are using batch weights, also update this value
            if self.batch:
                self.batch_weights = copy.deepcopy(self.weights)

    def evaluate_weights(self, eval_data, bigrams):
        e_data_fd = open(eval_data, 'r')
        w_eval_fd = open("./data/results.tsv", 'w')

        vector_eval = build_vector_models(e_data_fd, self.vocab, bigrams)
        for i, (y, X) in enumerate(vector_eval):
            predictions = calculate_outcome_scores(X, self.bias, self.weights)
            y_hat = (-10,0)
            for m, p in enumerate(predictions):
                if y_hat[0] < p:
                    y_hat = (p, m)
            w_eval_fd.write(str(y_hat[1])+"\n")

    def run(self, weights, bias):
        t = 0 # iternation number
        tracker = []
        while(t < 10000):
            # Do the "Double Shuffle" for good luck ~Jinho
            shuffle(self.vectors)
            shuffle(self.vectors)
            # The batch count
            if self.batch:
                b_count = 0
            # Iterate through the entire vector set
            for i, (y, X) in enumerate(self.vectors):
                # Calculate scores
                predictions = calculate_outcome_scores(X, bias, weights)
                # Determine y_hat
                y_hat = (-10,0)
                for m, p in enumerate(predictions):
                    if y_hat[0] < p:
                        y_hat = (p, m)
                # Continue to next iteration if y_hat and y are the same
                if y_hat[1] == y:
                    continue
                # The update dance - two paths: batch and not batch
                if not self.batch:
                    # Update weights normally
                    weights[y_hat[1]] = update_y_hat(weights[y_hat[1]], self.ETA, X)
                    weights[y] = update_y(weights[y], self.ETA, X)
                else:
                    # Update the batch weights INSTEAD of the "main" weights
                    self.batch_weights[y_hat[1]] = update_y_hat(weights[y_hat[1]], self.ETA, X)
                    self.batch_weights[y] = update_y(weights[y], self.ETA, X)
                    # If it has been 10 itterations of updates
                    if b_count is 10:
                        # Copy the values in the batch weights -> weights
                        weights = copy.deepcopy(self.batch_weights)
                    b_count += 1
                # Update bias
                bias[y_hat[1]] = bias[y_hat[1]] - self.ETA
                bias[y] = bias[y] + self.ETA
            # One rule I set for myself is that if you are done goign through
            # all of the vectors, weights == batch_weights
            if self.batch:
                weights = copy.deepcopy(self.batch_weights)
            # checks for convergence -------------------------------------------
            v = round(test_with_dev_data(weights, bias, self.vectors_dev), 7)
            tracker.append(v)
            if check_convergence(tracker):
                print v, t
                break
            t = t + 1

if __name__ == '__main__':
    Perceptron("./data/sst.train.tsv", "./data/sst.dev.tsv", use_bi_grams=True, batch=True, is_eval=True, eval_data="/Users/tshaban/Desktop/sst.tst.unlabeled.tsv")
