# !/bin/env python3
import json
import z3
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass 
from  functools import reduce 
from itertools import product
import networkx as nx 

EPSILON = z3.Const("epsilon", z3.RealSort())

class Graph:
    def __init__(self):
        self.nodes = set()
        self.adjacency_map = {}

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adjacency_map:
            self.adjacency_map[node] = []

# get the path between two nodes 
    def get_path(self, node1, node2):
        try:
            assert node1 in self.adjacency_map and node2 in self.adjacency_map
        except AssertionError as _:
            raise AssertionError("Not valid Nodes")
        vertex_queue = []
        path = []
        visited = []
        vertex_queue.append(node1)
        visited.append(node1)
        path.append(node1)
        while len(vertex_queue) > 0:
            v = vertex_queue.pop()
            succ_list = self.adjacency_map[v]
            for succ in succ_list:
                if succ not in visited:
                    visited.append(succ)
                    vertex_queue.append(succ)
                if succ == node2:
                    path.append(succ)
                    break 
                else:
                    path.append(succ)
        return path
                
    def add_edge(self, from_node, to_node):
        self.add_node(from_node)
        self.add_node(to_node)
        self.adjacency_map[from_node].append(to_node)
    def __str__(self):
        result = "Graph:\n"
        for node in sorted(self.nodes):
            result += f"{node}: {', '.join(map(str, self.adjacency_map.get(node, [])))}\n"
        return result

class AutomatonTransition:
    def __init__(self, from_state, to_state, formula):
        self.from_state = from_state
        self.to_state = to_state
        self.formula = formula

    def __str__(self):
        return f"From: {self.from_state}, To: {self.to_state}, Formula: {self.formula}"

class NodeAttributes:
    def __init__(self):
        self.alphabet = {}
        self.attribute_map = {}

    def add_variable(self, var_name, value):
        if isinstance(value, (int, float)):
            self.alphabet[var_name] = z3.Real(var_name)
        elif isinstance(value, str):
            self.alphabet[var_name] = z3.String(var_name)
        else:
            raise ValueError("Unsupported attribute type")


    def get_variable(self, var_name):
        return self.alphabet.get(var_name, None)
    
    def __str__(self):
        output = "Node Attributes:\n"
        for var_name, value in self.attribute_map.items():
            output += f"{var_name}: {value}\n"
        return output

class Automaton:
    def __init__(self):
        self.initial_state = None
        self.transitions = []
        self.final_states = set()
    def __str__(self):
        transitions_str = "\n".join(str(transition) for transition in self.transitions)
        return f"Initial State: {self.initial_state}, Transitions:\n{transitions_str}, Final States: {self.final_states}"



# state =  (aut_state, node)
@dataclass(frozen=True)
class ProductAut:
    initial_state: Tuple[str, str]
    final_state: Set[Tuple]
    transitions: Dict[Tuple, Tuple[str ,Tuple]]
    network: nx.Graph
    node: Set 

    def __str__(self) -> str:
        result = "Inite State:\n"
        result += ",".join(map(str, self.initial_state)) + "\n"
        result = "Finite State:\n"
        result += ",".join(map(str, self.final_state)) + "\n"
        result = "Transitions:\n"

        for node in sorted(self.transitions.keys()):
            result += f"{node}: {', '.join(map(str, self.transitions[node]))}\n"
        return result
       


def product_graph(aut:Automaton, graph:Graph):
    aut_state = set(map(
        lambda x: x.to_state, 
        aut.transitions
    ))
    trans  = {}
    aut_state.add(aut.initial_state)
    init_state = list((aut.initial_state, node) for node in graph.nodes)
    final_states = list(product(aut.final_states, graph.nodes))
    states = set(product(aut_state, graph.nodes))
    network = nx.MultiDiGraph()
    for transition in aut.transitions:
        from_nodes = set(filter(lambda x: x[0] == transition.from_state, states))
        for node in from_nodes:
            if node not in trans:
                trans[node] = []

            val = trans[node]
            trans[node] = val + [(transition.formula, (transition.to_state, to_node)) for to_node in graph.adjacency_map[node[1]]]
            for to_node in graph.adjacency_map[node[1]]:
                network.add_edge(node,(transition.to_state, to_node), transition.formula)
    return ProductAut(
        init_state,
        final_states,
        trans,
        network,
        graph.nodes
    )

def get_path(graph:ProductAut, node1, node2):
 
        vertex_queue = []
        path = []
        visited = []
        vertex_queue.append(node1)
        visited.append(node1)
        while len(vertex_queue) > 0:
            source = vertex_queue.pop()
            succ_list = graph.transitions[source]
            for transition, succ in succ_list:
                if succ not in visited:
                    visited.append(succ)
                    vertex_queue.append(succ)
                if succ == node2:
                    path.append((source,succ, transition ))
                    break 
                else:
                    path.append((source, succ, transition))
        return path
#######
###Every node in path (source, target, formula)
State = Tuple[int, int]
Transition = Tuple[State, State, str]


def find_all_path(graph: ProductAut, start:str, end:str):
        # try:
        #     assert start in graph.node and end in graph.node
        # except AssertionError as _:
        #     raise AssertionError("Not valid Nodes")
        path = []
        starts = set(filter(lambda x: x[1] == start, graph.initial_state))
        ends = set(filter(lambda x: x[1] == end, graph.final_state ))
        for start, end in product(starts, ends):
            p = sorted(nx.all_simple_edge_paths(graph.network, start, end))
            if len(p[0]) != 0 and len(p) ==1:
                path = path + p 
            else:
                p = get_path(graph,start, end)
                path.append(p) 
        return path


def naive_iter(attr:NodeAttributes, paths:List[List], vars):

    def substitute(formulas, vertex_attribute):
                curr = z3.parse_smt2_string(formulas,decls=vars)[0]
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

    formula_vector = []
    for path in paths:
        formula = []
        for source, target, f in path:
            source_state, source_node = source 
            attribute = attr.attribute_map[str(source_node)]
            curr_formula = substitute(f,attribute)
            formula.append(curr_formula)
        formula_vector.append(z3.And(formula))
    if len(formula_vector) == 1:
        return formula_vector[0]
    else:
        return z3.Or(formula_vector)


def naive_query(aut:Automaton, graph:Graph, para, source, target, attr:NodeAttributes):
    vars = merge_dicts(para, attr.alphabet)
    product_aut = product_graph(aut, graph)
    path = find_all_path(product_aut, source, target)
    if len(path) == 0:
        return z3.unsat
    formula = naive_iter(attr, path, vars)
    solver = z3.Solver()
    solver.add(formula)
    match solver.check():
        case z3.sat:
            return solver.model()
        case _:
            return solver.check()






VarBound = Dict[str, int]
@dataclass
class MacroState:
    vars: Dict 
    para_upper_bound: VarBound 
    para_lower_bound: VarBound 


def update_macro_state(
                       attr: NodeAttributes,  
                       macro:MacroState, 
                       transition: Transition, 
                       parameter
                       ) -> Optional[MacroState]:
                bound_vector =  z3.AstVector()
                source , target , formula = transition
                vertex_attribute = attr.attribute_map[str(source[1])]
                varset = list(filter(
                    lambda x: x in formula, 
                    parameter.keys()
                ))
                for var in varset:
                    if var in macro.para_lower_bound:
                        variable = parameter[var]
                        lower = macro.para_lower_bound[var]
                        bound_vector.push(variable>= lower)
                    if var in macro.para_upper_bound:
                        variable = parameter[var]
                        upper = macro.para_upper_bound[var]
                        bound_vector.push(variable<= upper)           

                curr = z3.parse_smt2_string(formula,decls=merge_dicts(parameter, attr.alphabet))[0]
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

                ################ Test the satisfiability ######3
                sat_solver = z3.Solver()
                if len(bound_vector) > 0:
                    sat_solver.add(bound_vector)
                sat_solver.add(curr)

                ####check if current state is sat with the constraint ####
                match sat_solver.check():
                    case z3.unsat:
                        return None 
                    case z3.sat:
                        pass 
                
                ### To solve the upper bound 
                upper_map = {}
                upper_bound_solver = z3.Optimize()
                if len(bound_vector) > 0:
                    upper_bound_solver.add(bound_vector)
                upper_bound_solver.add(curr)
                for para in varset:
                    var = parameter[para]
                    upper_map[para] = upper_bound_solver.maximize(var)
                upper_bound_solver.check()
                for para in varset:                    
                    var = upper_map[para]
                    value = upper_bound_solver.upper(var)                    
                    value = z3.substitute(value, (EPSILON, z3.RealVal(0.000001)))
                    value = z3.simplify(value)
                    if isinstance(value, z3.RatNumRef):
                        macro.para_upper_bound[para] = value


                    

                ### To solve the lower bound
                lower_map = {}
                lower_bound_solver = z3.Optimize()
                if len(bound_vector) > 0:
                    lower_bound_solver.add(bound_vector)
                lower_bound_solver.add(curr)
                for para in varset:
                    var = parameter[para]
                    lower_map[para] = lower_bound_solver.minimize(var)
                lower_bound_solver.check()
                for para in varset:
                    var = lower_map[para]
                    value = lower_bound_solver.lower(var)
                    value = z3.substitute(value, (EPSILON, z3.RealVal(0.000001)))
                    value = z3.simplify(value)
                    if isinstance(value, z3.RatNumRef):
                        macro.para_lower_bound[para] = value
                        

                #MODIFY STATE 
                return macro

def explore_one_path(path:List[Transition],
                             attr:NodeAttributes,
                             macro_state:MacroState, 
                             parameter):
    for transition in path:
        macro_state = update_macro_state(attr,macro_state, transition, parameter)
        if macro_state is None:
            return z3.unsat 
    bound = []
    for var in parameter.keys():
        if var in macro_state.para_lower_bound:
            lower = macro_state.para_lower_bound[var].as_decimal(3)
        else: 
            lower =  ()
        if var in macro_state.para_upper_bound:
            upper = macro_state.para_upper_bound[var].as_decimal(3)
        else:
            upper = ()
        bound.append((var,lower,upper))
    return bound 

def query_with_macro_state(graph:Graph, attr:NodeAttributes, aut:Automaton, source, target, parameter) -> bool:
    upper_bound = {}
    lower_bound = {} 
    macro_state = MacroState(
      merge_dicts(attr.alphabet, parameter), 
      upper_bound,
      lower_bound      
  )
    pg = product_graph(aut,graph)
    paths = find_all_path(pg,source, target)
    for path in paths:
        result =  explore_one_path(path, attr, macro_state, parameter)
        if result == z3.unsat:
            pass 
        else:
            return result 
    return z3.unsat






def create_global_var(var_name, type):
        if type == "Real":
            return z3.Real(var_name)
        elif type == "String":
            return z3.String(var_name)
        else:
            raise ValueError("Unsupported attribute type")

def merge_dicts(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Key(s) {common_keys} are present in both dictionaries and would be overwritten")

    return {**dict1, **dict2}

def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    graph_db = Graph()
    attributes = NodeAttributes()
    automaton = Automaton()
    attribute_map = {}
    global_vars = {}

    # Parse Graph Database
    for edge in json_data["Graph Database"]["Edges"]:
        from_node, to_node = map(int, edge.split(" -> "))
        graph_db.add_edge(from_node, to_node)

    # Parse Attributes
    for vertex, attr in json_data["Attributes"].items():
        for attr_name, attr_value in attr.items():
            attributes.add_variable(attr_name, attr_value)

    for vertex, attr in json_data["Attributes"].items():
        attr_tuple = tuple(attr.values())
        attribute_map[vertex] = attr_tuple
    attributes.attribute_map = attribute_map
    # Parse Automaton
    automaton.initial_state = json_data["Automaton"]["Initial State"]
    automaton.final_states = set(json_data["Automaton"]["Final States"])
    automaton.transitions = [
        AutomatonTransition(t['from'], t['to'], t['formula']) for t in json_data["Automaton"]["Transitions"]
    ]

    # Parse Global Variables
    for name, type in json_data["Global Variables"].items():
        global_vars[name] = create_global_var(name, type)

    return graph_db, attributes, automaton, global_vars