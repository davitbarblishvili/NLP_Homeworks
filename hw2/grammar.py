"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum
import math


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        for _, grammar_rules in self.lhs_to_rules.items():
            for rule in grammar_rules:
                rhs = rule[1]
                if len(rhs) == 2:
                    if rhs[0].upper() != rhs[0] and rhs[1].upper() != rhs[1]:
                        print(
                            f"The grammar can not be verified. {rhs[0]} and {rhs[1]} have to be both non terminal values")
                        return False

                elif len(rhs) == 1:
                    if rhs[0].lower() != rhs[0]:
                        print(
                            f"The grammar can not be verified. {rhs[0]} has to be terminal value")
                        return False
                else:
                    return False
            prob_sum = fsum(rule[2] for rule in grammar_rules)
            if abs(1 - prob_sum) > 0.00001:
                print(prob_sum)
                print(f"The probability of the same lhs symbol does not add up to 1")
                return False
        return True


if __name__ == "__main__":

    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

        if grammar.verify_grammar():
            print("Grammar is successfully verified. Grammar is a valid PCFG in CNF")
        else:
            print(
                "Grammar is not successfully verified. Grammar is not a valid PCFG in CNF")
