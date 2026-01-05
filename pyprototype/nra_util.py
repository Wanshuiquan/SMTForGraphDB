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
        pre_cond = z3.BoolVal(True)
        for f in self.accumulated_formula:
            pre_cond = z3.And(pre_cond, f)
        solver.add(z3.Implies(pre_cond, other))
        res = solver.check() == z3.sat 
        solver.pop()
        return res
    

    def check_consistency(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        pre_cond = z3.BoolVal(True)
        for f in self.accumulated_formula:
            pre_cond = z3.And(pre_cond, f)
        solver.add(z3.And(pre_cond, other))
        result = solver.check() == z3.sat
        solver.pop()
        return result
    

    def visit_new_constraint(self, new_constraint: z3.Ast, solver:z3.Solver) -> bool:
        if self.entail(new_constraint, solver):
            return True
        elif self.check_consistency(new_constraint, solver):
                self.accumulated_formula.add(new_constraint)     
        else:
            return False
        

@dataclass
class NRA_Mavro_State_Optimizer:
    accumulated_formula: Set[z3.Ast]
    automata_state: int  
    edge_id: int  
    node_id: int 
    globalparam: Dict[str, z3.AST]

    def entail(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        pre_cond = z3.BoolVal(True)
        for f in self.accumulated_formula:
            pre_cond = z3.And(pre_cond, f)
        solver.add(z3.Implies(pre_cond, other))
        res = solver.check() == z3.sat 
        solver.pop()
        return res
    

    def check_consistency(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        pre_cond = z3.BoolVal(True)
        for f in self.accumulated_formula:
            pre_cond = z3.And(pre_cond, f)
        solver.add(z3.And(pre_cond, other))
        result = solver.check() == z3.sat
        solver.pop()
        return result
    
    def optimize_parameters(self, optimizer: z3.Optimize):
        for param_name in self.globalparam.keys():
            optimizer.minimize(self.globalparam[param_name])
        optimizer.check()
        optimizer.model()
        
    def visit_new_constraint(self, new_constraint: z3.Ast, solver:z3.Solver, optimizer:z3.Optimize) -> bool:
        if self.entail(new_constraint, solver):
            return True
        elif len(self.accumulated_formula) < len(self.globalparam):
            if self.check_consistency(new_constraint, solver):
                self.accumulated_formula.add(new_constraint)    
        else:
            pass 
            

        

def merge_dicts(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Key(s) {common_keys} are present in both dictionaries and would be overwritten")

    return {**dict1, **dict2}
          


class ProductGraphIter:
    def __init__(self, prop_graph:PropertyGraph, automaton:ParametricAutomaton):
            self.automata = automaton
            self.graph = prop_graph 
            self.solver = z3.Solver()
            self.alphabet = {
                "since": z3.Real("since"),
                "age": z3.Real("age")
            }
            self.globalparam = {}
            for var_name in self.automata.global_parameters:
                self.globalparam[var_name] = z3.Real(var_name)


    def explore(self, start_point: int) -> bool:
        start_state = NRA_Macro_State(
            accumulated_formula = set(),
            automata_state = self.automata.initial_state,
            edge_id = -1,
            node_id = start_point
        )
        if self.automata.initial_state in self.automata.accepting_states:
            return True 
        frontier = [start_state]
        explored = set()
        while frontier:
            current_macro_state = frontier.pop()
            if (current_macro_state.automata_state, current_macro_state.node_id) in explored:
                continue
            explored.add((current_macro_state.automata_state, current_macro_state.node_id))
            flag, neighbors = self.expand_neighbors(current_macro_state)
            if flag:
                return True
            frontier.extend(neighbors)
        return False
    
        
    def expand_neighbors(self, current_macro_state: NRA_Macro_State):
        automata_loc = current_macro_state.automata_state
        graph_loc = current_macro_state.node_id
        neighbors = []
        automata_dsts = self.automata.transitions[automata_loc] 
        if graph_loc not in self.graph.edges.keys():
            return False, neighbors
        graph_dsts = self.graph.edges[graph_loc]
        for dst_state in automata_dsts.keys():
            (edge_label_a, constraint) = automata_dsts[dst_state]
            for (edge_label_g, graph_dst) in graph_dsts:
                if edge_label_a == edge_label_g:
                    new_formula = self.substitute(constraint, self.graph.attribute[graph_dst])
                    new_macro_state = NRA_Macro_State(
                        accumulated_formula = current_macro_state.accumulated_formula.copy(),
                        automata_state = dst_state,
                        edge_id = edge_label_a,
                        node_id = graph_dst
                    )
                    if new_macro_state.visit_new_constraint(new_formula, self.solver):
                        neighbors.append(new_macro_state)
                        if dst_state in self.automata.accepting_states:
                            return True, neighbors
        return False, neighbors
    


    def substitute(self, formulas, vertex_attribute):
                curr = z3.parse_smt2_string(formulas,decls=merge_dicts(self.alphabet, self.globalparam))[0]
                for attribute in self.alphabet.keys():
                
                    if attribute in vertex_attribute:
                        var_name = self.alphabet[attribute]
                        val = vertex_attribute[attribute]
                        if isinstance(val, str):
                           curr = z3.substitute(curr,(var_name, z3.StringVal(val)))
                        else:
                            curr = z3.substitute(curr,(var_name, z3.RealVal(val)))
                return curr 
    
def query_property_graph(prop_graph:PropertyGraph, automaton:ParametricAutomaton, start_node) -> bool:
    product_iter = ProductGraphIter(prop_graph, automaton)
    for start_node in prop_graph.nodes:
        if product_iter.explore(start_node):
            return True 
    return False