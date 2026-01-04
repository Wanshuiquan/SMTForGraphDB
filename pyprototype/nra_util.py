from dataclasses import dataclass 
from typing import Set, Dict, Any, Tuple 
import z3
import json 
from itertools import product
from property_graph import PropertyGraph
from parametric_automata import ParametricAutomaton


EPSILON = z3.Const("epsilon", z3.RealSort())

@dataclass 
class NRA_Macro_State:
    accumulated_formula: Set[z3.Ast]
    automata_state: int  
    edge_id: int  
    node_id: int 

    def entail(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        solver.add(z3.Implies(z3.And(list(self.accumulated_formula)), other))
        res = solver.check() == z3.sat 
        solver.pop()
        return res
    

    def check_consistency(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        solver.add(z3.And(list(self.accumulated_formula), other))
        result = solver.check() == z3.sat
        solver.pop()
        return result
    

    def visit_new_constraint(self, new_constraint: z3.Ast) -> bool:
        if self.entail(new_constraint):
            return True
        elif self.check_consistency(new_constraint):
            self.accumulated_formula.add(new_constraint)
            return True 
        else:
            return False
        




def merge_dicts(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Key(s) {common_keys} are present in both dictionaries and would be overwritten")

    return {**dict1, **dict2}
          
@dataclass
class ProductGraph:
    nodes: Set[tuple[int, int]]  # (property_node_id, automaton_state)
    edges: Dict[tuple[tuple[int, int], tuple[str, z3.Ast, tuple[int, int]]]]  # ((src_property_node_id, automaton_state), edge_label, (tgt_property_node_id, automaton_state))
    

    def __init__(self, prop_graph: PropertyGraph, automaton: ParametricAutomaton):
        self.nodes = set()
        self.edges = dict()
        self.construct_product(prop_graph, automaton)


    def construct_product(self, prop_graph: PropertyGraph, automaton: ParametricAutomaton):
        for prop_node, automaton_state in product(prop_graph.nodes, automaton.transitions):
            self.nodes.add((prop_node, automaton_state))
            automaton_dsts = automaton.transitions[automaton_state] 
            graph_dsts = prop_graph.edges[prop_node]
            for (edge_label_a, constraint), automaton_dst in automaton_dsts:
                for edge_label_g, graph_dst in graph_dsts:
                    if edge_label_a == edge_label_g:
                        self.edges[(prop_node, automaton_state)] = (edge_label_a, constraint, (graph_dst, automaton_dst))            



class ProductGraphIter:
    def __init__(self, prop_graph:PropertyGraph, automaton:ParametricAutomaton):
            self.automata = automaton
            self.graph = prop_graph 
            self.product = ProductGraph(prop_graph, automaton)
            self.solver = z3.Solver()
            self.alphabet = {}
            for var_name in self.automata.global_parameters:
                self.alphabet[var_name] = z3.Real(var_name)
            self.alphabet["since"] = z3.Real("since")
            self.alphabet["age"] = z3.Real("age")
    
    def explore(start):
        pass
    def substitute(self, formulas, vertex_attribute):
                curr = z3.parse_smt2_string(formulas,decls=self.alphabet)[0]
                keys = list(attr.alphabet.keys())
                for index in range(len(keys)):
                    attribute = keys[index]
                    if vertex_attribute[index] != None:
                        var_name = attr.alphabet[attribute]
                        val = vertex_attribute[index]
                        if isinstance(val, str):
                           curr = z3.substitute(curr,(var_name, z3.StringVal(val)))
                        else:
                            curr = z3.substitute(curr,(var_name, z3.RealVal(val)))
                return curr 