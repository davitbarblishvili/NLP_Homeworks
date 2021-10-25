"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###


def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write(
                "Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write(
                "Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write(
                "Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write(
                "Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write(
                    "Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar.
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if
        the sentence is in the language described by the grammar. Otherwise
        return False

        """
        rules = list(self.grammar.rhs_to_rules)
        pi_table = [[[] for i in range(len(tokens)+1)]
                    for j in range(len(tokens)+1)]

        for i in range(len(tokens)):
            for tag, grammar_rules in self.grammar.rhs_to_rules.items():
                if grammar_rules[0][1] == (tokens[i],):
                    pi_table[i][i+1] = set()
                    for item in self.grammar.rhs_to_rules[grammar_rules[0][1]]:
                        pi_table[i][i+1].add(item[0])

        for length in range(2, len(tokens) + 1):
            for i in range(len(tokens) - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    if not pi_table[i][j]:
                        pi_table[i][j] = set()
                    for _, grammar_rules in self.grammar.rhs_to_rules.items():
                        if len(grammar_rules[0][1]) == 2:
                            for tag_1 in pi_table[i][k]:
                                for tag_2 in pi_table[k][j]:
                                    if tag_1 == grammar_rules[0][1][0] and tag_2 == grammar_rules[0][1][1]:
                                        for item in self.grammar.rhs_to_rules[grammar_rules[0][1]]:
                                            pi_table[i][j].add(item[0])

        if pi_table[0][len(tokens) - 1]:
            return True

        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = dict()
        probs = dict()
        len_tokens = len(tokens)
        # table dict initialization
        for i in range(len_tokens+1):
            for j in range(len_tokens+1):
                table[(i, j)] = dict()

        # prob dict initialization
        for i in range(len_tokens+1):
            for j in range(len_tokens+1):
                probs[(i, j)] = dict()

        for i in range(len_tokens):
            for _, grammar_rules in self.grammar.rhs_to_rules.items():
                if grammar_rules[0][1] == (tokens[i],):
                    table[(i, i+1)] = dict()
                    probs[(i, i+1)] = dict()
                    for item in self.grammar.rhs_to_rules[grammar_rules[0][1]]:
                        table[(i, i+1)][item[0]] = item[1][0]
                        probs[(i, i+1)][item[0]] = math.log(item[2])

        for length in range(2, len_tokens + 1):
            for i in range(len_tokens - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    if (i, j) not in table.keys():
                        table[(i, j)] = dict()
                        probs[(i, j)] = dict()
                for _, grammar_rules in self.grammar.rhs_to_rules.items():
                    if len(grammar_rules[0][1]) == 2:
                        for tag_1 in table[(i, k)].keys():
                            for tag_2 in table[(k, j)].keys():
                                if tag_1 == grammar_rules[0][1][0] and tag_2 == grammar_rules[0][1][1]:
                                    for item in self.grammar.rhs_to_rules[grammar_rules[0][1]]:
                                        if item[2] != 0:
                                            items = ((tag_1, i, k),
                                                     (tag_2, k, j))
                                            prob = math.log(
                                                item[2]) + probs[(i, k)][tag_1] + probs[(k, j)][tag_2]
                                            if item[0] not in probs[(i, j)]:
                                                probs[(i, j)][item[0]] = prob
                                                table[(i, j)][item[0]] = items
                                            else:
                                                current_prob = probs[(
                                                    i, j)][item[0]]
                                                if prob > current_prob:
                                                    probs[(i, j)][item[0]
                                                                  ] = prob
                                                    table[(i, j)][item[0]
                                                                  ] = items
                                        else:
                                            current_prob = 0
                                            if item[0] not in table[(i, j)]:
                                                table[(i, j)][item[0]] = (
                                                    (tag_1, i, k), (tag_2, k, j))
                                                probs[(i, j)][item[0]
                                                              ] = current_prob
        return table, probs

    def parse_with_backpointers1(self, tokens):

        rules = list(self.grammar.rhs_to_rules)
        n = len(tokens)
        table = dict()
        probs = dict()
        for i in range(n+1):
            for j in range(n+1):
                table[(i, j)] = dict()
        for i in range(n+1):
            for j in range(n+1):
                probs[(i, j)] = dict()

        for i in range(n):
            for rule in rules:
                if rule == (tokens[i],):
                    table[(i, i+1)] = dict()
                    probs[(i, i+1)] = dict()
                    for x in self.grammar.rhs_to_rules[rule]:
                        table[(i, i+1)][x[0]] = x[1][0]
                        probs[(i, i+1)][x[0]] = math.log(x[2])
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i+length
                for k in range(i+1, j):
                    if (i, j) not in table.keys():
                        table[(i, j)] = dict()
                        probs[(i, j)] = dict()
                    for rule in rules:
                        if len(rule) == 2:
                            #print(rule, table[(i, k)],table[(k,j)])
                            for symbol1 in table[(i, k)].keys():
                                for symbol2 in table[(k, j)].keys():
                                    if symbol1 == rule[0] and symbol2 == rule[1]:
                                        for x in self.grammar.rhs_to_rules[rule]:
                                            if not x[2] == 0:
                                                children = (
                                                    (symbol1, i, k), (symbol2, k, j))
                                                probabiity = math.log(
                                                    x[2]) + probs[(i, k)][symbol1] + probs[(k, j)][symbol2]
                                                if x[0] not in probs[(i, j)]:
                                                    probs[(i, j)][x[0]
                                                                  ] = probabiity
                                                    table[(i, j)][x[0]
                                                                  ] = children
                                                else:
                                                    current_prob = probs[(
                                                        i, j)][x[0]]
                                                    if probabiity > current_prob:
                                                        probs[(i, j)][x[0]
                                                                      ] = probabiity
                                                        table[(i, j)][x[0]
                                                                      ] = children
                                            else:
                                                current_prob = 0
                                                if x[0] not in table[(i, j)]:
                                                    table[(i, j)][x[0]] = (
                                                        (symbol1, i, k), (symbol2, k, j))
                                                    probs[(i, j)][x[0]
                                                                  ] = current_prob
        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    return None


if __name__ == "__main__":

    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)

        assert check_table_format(table)
        assert check_probs_format(probs)
