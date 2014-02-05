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

class Trainer:
    def __init__(self):
        self.tags = set()
        self.f = set()
        self.v = {}
        self.x = []
        self.y = []

    def read_training_data(self, training_file):
        """
        """
        if debug: sys.stdout.write("Reading training data...\n")

        file = open(training_file, 'r')
        file.seek(0)
        sentence = []
        tags = []
        for line in file:
            line = line.strip()
            if line: # Non-empty line
                token = line.split()
                word = token[0]
                tag  = token[1]
                sentence.append(word)
                tags.append(tag)
            else: # End of sentence reached
                self.x.append(tuple(sentence))
                self.y.append(tuple(tags))
                self.tags.update(tags)
                sentence = []
                tags = []
        file.close()

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

    def reset_weights(self):
        """
        Reset all the weights to zero in the weight vector.
        """
        for feature in self.v:
            self.v[feature] = 0

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

    def perceptron_algorithm(self, iterations):
        """
        Run the perceptron algorithm to estimate (or improve) the weight vector.
        """
        self.reset_weights()

        for iteration in range(iterations):
            if debug: sys.stdout.write("Perceptron algorithm iteration %d...\n" % (iteration+1))

            # Loop over each sentence/tag pair (xi, yi) in the training data
            for i in range(len(self.x)):
                x = list(self.x[i])
                y = list(self.y[i])

                # Find the best tagging sequence using the Viterbi algorithm
                prob, z = self.viterbi_algorithm(x)

                # Check if the tags match the gold standard
                if z != y:

                    # Compute the gold tagging feature vector f(x, y)
                    # and the best tagging feature vector f(x, z)
                    fy = {}
                    fz = {}

                    # Add the start and stop symbols to the tagging sequences
                    x = x + [STOP]
                    y = [START, START] + y + [STOP]
                    z = [START, START] + z + [STOP]

                    # Loop over all words in the sentence
                    for k in range(len(x)):

                        # Calculate the features for the next gold tag (y_k)
                        f = self.feature_vector(dict(t2=y[k], t1=y[k+1], x=x, i=k+1), y[k+2])

                        # Add the features to the gold tagging feature vector
                        for feature in f:
                            fy.setdefault(feature, 0)
                            fy[feature] += 1

                        # Calculate the features for the next best tag (z_k)
                        f = self.feature_vector(dict(t2=z[k], t1=z[k+1], x=x, i=k+1), z[k+2])

                        # Add the features to the best tagging feature vector
                        for feature in f:
                            fz.setdefault(feature, 0)
                            fz[feature] += 1

                    # Update the feature vector (adding new features if necessary)
                    for feature in fy:
                        self.v.setdefault(feature, 0)
                        self.v[feature] += fy[feature]
                    for feature in fz:
                        self.v.setdefault(feature, 0)
                        self.v[feature] -= fz[feature]

            if debug:
                sys.stdout.write("Updated weight vector:\n")
                self.write_weight_vector()

    def write_weight_vector(self):
        """
        """
        for feature in self.v:
            sys.stdout.write("%s %.1f\n" % (feature, self.v[feature]))

def main(training_file):
    """
    """

    trainer = Trainer()

    # Read the training data (x, y)
    trainer.read_training_data(training_file)

    # Compute the weight vector using the Perceptron algorithm
    trainer.perceptron_algorithm(5)

    # Write out the final weight vector
    trainer.write_weight_vector()

def usage():
    sys.stderr.write("""
    Usage: python perceptron_training.py [training_file]
        Find the most likely tag sequence for each sentence in the input
        data file using a Global Linear Model decoder with a previously
        determined weight vector.\n""")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
    main(sys.argv[1])
