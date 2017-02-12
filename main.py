import random
from random import shuffle

# Build n_grams for the model
def stich_ngrams(n_grams, n):
    grams = []
    for g in n_grams:
        s = ""
        for w in g:
            s = s + w + "_"
        grams.append(s[0:len(s)-1])
    return grams

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
def build_vocab(fd):
    index = 0
    vocab = {}
    for line in fd:
        words = line.split()

        # Add bi-grams
        bi_grams = find_ngrams(words[1:], 2)
        for gram in bi_grams:
            for word in gram:
                lower_case_word = word.lower()
                if not vocab.has_key(lower_case_word):
                    vocab[lower_case_word] = index
                    index = index + 1

        # Add uni-grams
        for word in words[1:]:
            lower_case_word = word.lower()
            if not vocab.has_key(lower_case_word):
                vocab[lower_case_word] = index
                index = index + 1
    return vocab, index

def build_vector_models(fd, vocab):
    vectors = []
    for line in fd:
        words = line.split()
        dic = {}
        tag = int(words[0])

        # Add bi-grams
        bi_grams = find_ngrams(words[1:], 2)
        for gram in bi_grams:
            for word in gram:
                lower_case_word = word.lower()
                if dic.has_key(lower_case_word):
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
# Set up perceptron -----------------------------------
fd = open("sst.train.tsv", 'r')
# Vocab
vocab_lookup_dict, size = build_vocab(fd)

fd = open("sst.train.tsv", 'r')
# Array of Vectors
vectors = build_vector_models(fd, vocab_lookup_dict)
print(size)
# Weights
weights = build_initial_weights(size)
# Bias
bias = [1 for i in range(5)]
# Learning rate
ETA = 0.1
print ETA
# Open DEV data set for test
f_t = open("sst.dev.tsv", 'r')
vectors_dev = build_vector_models(f_t, vocab_lookup_dict)

# Run perceptron --------------------------------------
t = 0 # iternation number

c = []
while(t < 100):
    # Do the "Double Shuffle" for good luck #Jinho
    # vectors = shuffle(shuffle(vectors))
    correct = 0.0
    shuffle(vectors)
    shuffle(vectors)
    for i, (y, X) in enumerate(vectors):
        # Calculate scores
        predictions = calculate_outcome_scores(X, bias, weights)
        # Determine y_hat
        y_hat = (-10,0)
        for m, p in enumerate(predictions):
            if y_hat[0] < p:
                y_hat = (p, m)
        # Continue to next iteration if y_hat and y are the same
        if y_hat[1] == y:
            correct = correct + 1.0
            continue
        weights[y_hat[1]] = update_y_hat(weights[y_hat[1]], ETA, X)
        weights[y] = update_y(weights[y], ETA, X)
        bias[y_hat[1]] = bias[y_hat[1]] - ETA
        bias[y] = bias[y] + ETA
        # track[i][t%20] = y_hat[1]
    c.append(correct)
    t = t + 1

# for i in c:
#     print i, len(vectors)

# Test perceptron --------------------------------------
count_correct = 0

for i, (y, X) in enumerate(vectors_dev):
    # Calculate scores
    predictions = calculate_outcome_scores(X, bias, weights)
    # Determine y_hat
    y_hat = (-10,0)
    for m, p in enumerate(predictions):
        if y_hat[0] < p:
            y_hat = (p, m)
    # Continue to next iteration if y_hat and y are the same
    if y_hat[1] == y:
        count_correct = count_correct + 1

print count_correct, len(vectors_dev)
