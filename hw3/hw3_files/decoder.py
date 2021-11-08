from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import numpy as np
import keras
import tensorflow as tf
from extract_training_data import FeatureExtractor, State


class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = tf.keras.models.load_model(modelfile)
        self.extractor = extractor
        self.dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc',
                              'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg', 'dt', 'det']

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (
            action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:

            representation_ = self.extractor.get_input_representation(
                words, pos, state)
            possible_moves = self.model(
                np.array([representation_]), training=False)
            sorted_indices = reversed(np.argsort(possible_moves)[
                0])

            allowed_operations = list(filter(legality(state), np.arange(91)))

            # filter allowed operations
            legal_transition_index = list(
                filter(contain(allowed_operations), sorted_indices))
            # pick out the highest scoring permitted transition
            transition_index = legal_transition_index[0]

            if transition_index < 45:
                (operator, label) = ("left_arc",
                                     self.dep_relations[transition_index])
            elif transition_index < 90:
                (operator,
                 label) = ("right_arc", self.dep_relations[transition_index - 45])
            else:
                (operator, label) = ("shift", None)

            if operator == "left_arc":
                state.left_arc(label)
            if operator == "right_arc":
                state.right_arc(label)
            if operator == "shift":
                state.shift()

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


def legality(state):
    def if_legal(transition_index):
        # arc-left or arc-right are not permitted the stack is empty
        if len(state.stack) == 0:
            if transition_index < 90:
                return False
        # shifting the only word out of the buffer is also illegal, when the stack is not empty
        if (len(state.stack) > 0) and (len(state.buffer) == 1):
            if transition_index == 90:
                return False
        # the root node must never be the target of a left-arc
        if (len(state.stack) > 0) and (state.stack[-1] == 0):
            if transition_index < 45:
                return False
        return True
    return if_legal


def contain(allowed_operations):
    def if_contain(sorted_indices):
        return sorted_indices in allowed_operations
    return if_contain


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(
            WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
