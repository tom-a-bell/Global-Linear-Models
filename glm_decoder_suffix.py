#! /usr/bin/python

__author__="Tom Bell <tom.bell.code@gmail.com>"
__date__ ="$May 12, 2013"

import sys

"""
"""

# Debug output flag
debug = False

START = '*'
STOP  = 'STOP'

TRIGRAM = 'TRIGRAM'
TAG = 'TAG'
SUFFIX = 'SUFF'

class Decoder:
    def __init__(self):
        self.tags = set()
        self.f = set()
        self.v = {}

    def read_weights(self, weight_file):
        """
        Read the previously determined weight vector
        features and values from the input file.
        """
        if debug: sys.stdout.write("Reading weight vector...\n")

        file = open(weight_file, 'r')
        file.seek(0)
        for line in file:
            token = line.split()
            feature = token[0]
            weight  = float(token[1])
            self.f.add(feature)
            self.v[feature] = weight

            # Add the tags from any trigram features to the list of tags
            token = feature.split(':')
            if token[0] == TRIGRAM:
                self.tags.update(token[1:])
        file.close()

        # Remove the start and stop symbols from the list of tags
        self.tags.remove(START)
        self.tags.remove(STOP)

    def feature_vector(self, history, tag):
        """
        Compute the feature vector g(h, t) for the given history/tag pair.
        """
        # Store the feature vector as a map from feature strings to counts.
        g = {}

        # Generate the possible features to search
        f = set()

        # Trigram feature
        f.add(TRIGRAM+':'+history['t2']+':'+history['t1']+':'+tag)

        # Tag feature (if not beyond the end of the sentence)
        if history['i'] <= len(history['x']):
            f.add(TAG+':'+history['x'][history['i'] - 1]+':'+tag)

        # Suffix features (if not beyond the end of the sentence)
        if history['i'] <= len(history['x']):
            for j in range(1,3):
                word = history['x'][history['i'] - 1]
                if j <= len(word):
                    f.add(SUFFIX+':'+word[-j:]+':'+str(j)+':'+tag)

        # Check each feature
        for feature in f:
            token = feature.split(':')

            # Calculate trigram features
            if token[0] == TRIGRAM:
                s = token[1]
                u = token[2]
                v = token[3]
                if history['t2'] == s and history['t1'] == u and tag == v:
                    g[feature] = 1

            # Calculate tag features
            if token[0] == TAG:
                # Handle cases where ':' appears as part of the word
                r = ':'.join(token[1:-1])
                u = token[-1]
                if history['i'] <= len(history['x']):
                    w = history['x'][history['i'] - 1]
                else:
                    w = ''
                if w == r and tag == u:
                    g[feature] = 1


            # Calculate suffix features
            if token[0] == SUFFIX:
                # Handle cases where ':' appears as part of the word
                u = ':'.join(token[1:-2])
                j = int(token[-2])
                v = token[-1]
                if history['i'] <= len(history['x']):
                    w = history['x'][history['i'] - 1]
                else:
                    w = ''
                if w[-j:] == u and tag == v:
                    g[feature] = 1

        return g

    def inner_product(self, g):
        """
        Compute the inner product v.g(h, t)
        """
        return sum((self.v.get(key, 0) * value for key, value in g.iteritems()))

    def viterbi_algorithm(self, x):
        """
        Run the GLM form of the Viterbi algorithm on the input sentence (x)
        using the previously determined feature (f) and weight (v) vectors.
        """
        n = len(x)

        # Initialise pi and bp
        pi = {}
        bp = {}
        pi[(0, '*', '*')] = 0

        for k in range(1, n+1):

            if k == 1:
                T = {'*'}
                U = {'*'}
                S = self.tags
            elif k == 2:
                T = {'*'}
                U = self.tags
                S = self.tags
            else:
                T = self.tags
                U = self.tags
                S = self.tags

            for u in U:
                for s in S:
                    pi[(k, u, s)], bp[(k, u, s)] = max(((pi[(k-1, t, u)] + self.inner_product(self.feature_vector(dict(t2=t, t1=u, x=x, i=k), s)), t) for t in T))

        # Store the tag sequence as an array
        tag = ['']*(n+1)

        # Calculate the tag sequence by following the back pointers
        prob, tag[n-1], tag[n] = max((max(((pi[(n, u, s)] + self.inner_product(self.feature_vector(dict(t2=u, t1=s, x=x, i=n+1), STOP)), u, s) for s in S)) for u in U))
        for k in range(n-2, 0, -1):
            tag[k] = bp[(k+2, tag[k+1], tag[k+2])]
        tag = tag[1:]

        # Return the probability and tag sequence
        return prob, tag

    def tag_words(self, data_file):
        """
        Read each sentence from the input data file and generate a tag sequence.
        """
        sentence = []
        file = open(data_file, 'r')
        file.seek(0)
        for line in file:
            word = line.strip()
            if word: # Non-empty line
                sentence.append(word)
            else: # End of sentence reached
                prob, tag = self.viterbi_algorithm(sentence)
                for n in range(len(sentence)):
                    sys.stdout.write("%s %s\n" % (sentence[n], tag[n]))
                sys.stdout.write("\n")
                sentence = []
        file.close()

def main(weight_file, data_file):
    """
    """

    decoder = Decoder()

    # Read the previously determined weight vector
    decoder.read_weights(weight_file)

    # Find the most likely tag sequence for each sentence in the data file
    decoder.tag_words(data_file)

def usage():
    sys.stderr.write("""
    Usage: python glm_decoder.py [weight_file] [data_file]
        Find the most likely tag sequence for each sentence in the input
        data file using a Global Linear Model decoder with a previously
        determined weight vector.\n""")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
