from dataclasses import dataclass 
from typing import Set, Dict, Any, Tuple 
import z3
from itertools import product
import networkx as nx 
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
        

@dataclass 
class ParametricAutomaton:
    initial_state: int
    accepting_states: Set[int]
    transitions: Dict[int, Tuple[Tuple[str, z3.Ast], int]]  # source_state -> ((edge_label, constraint), target_state)


@dataclass
class PropertyGraph:
    edges: Dict[int, Tuple[Tuple[str, int]]]  # source_node ->  label, target_node)
    nodes: Set[int] # (node_id, node_type)
    attribute: Dict[Tuple[int, str], Any]  #node_id, attribute_name -> attribute_value

def generate_property_graph(edge_num:int, node_num:int):
      """
      Docstring for generate_property_graph
      
      :param edge_num: Description
      :type edge_num: int
      :param node_num: Description
      :type node_num: int
      """

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
