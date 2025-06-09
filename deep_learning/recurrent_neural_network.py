vocabulary_size = 8000
import csv
import itertools

import nltk

unknown_token = "UNKNOWN_TOKEN"

sentence_start_token = "SENTENCE_START"

sentence_end_token = "SENTENCE_END"

print ("Reading CSV file...")




with open ('data/reddit-comments-2015-08.csv','rb') as f:

    reader = csv.reader(f, skipinitialspace = True)
    reader.next()
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    
    sentences = sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print ("Parsed %d sentences." % (len(sentences)))


tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]


word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

vocab = word_freq.most_common(vocabulary_size-1)

index_to_word = [x[0] for x  in vocab]

index_to_word.append(unknown_token)

word_to_index = dict([(w,i) for i, w in enumerate(index_to_word)])


print("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))




# st = tanh(Ux_t + W s_t-1)

# o_t = softmax(Vs_t)

# W = 100 x 100

# U = 100 x 8000

# V = 8000 x 100
import numpy as np

class RNN:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):


        self.word_dim = word_dim
        self.hidden_dm = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_prop(self, x):

        T = len(x)

        s = np.zeros((T+1, self.hidden_dim))

        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]]) + self.W.dot(s[t-1])
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]
    

    def softmax(self, x):

        def __call__(x):
            e_x = np.exp(x-np.max(x, axis = -1, keepdims = True))
        
        def gradient(x):




