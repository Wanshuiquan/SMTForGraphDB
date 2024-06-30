# !/bin/env python3
from itertools import product
import json
import z3
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass 
from  functools import reduce
import networkx as nx
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




####Naive Recursive Algorithm #####
# explore path, state   
#         nil   state ::=  is_final_state(nil)
#        cons v::p state ::=  curr(v, state) and explore()
def explore_path(path:List[int],state, attr:NodeAttributes, aut:Automaton, var_dict):
        def substitute(formulas, vertex_attribute):
                curr = z3.parse_smt2_string(formulas,decls=var_dict)[0]
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
        if len(path) == 0:
            acc = state in aut.final_states 
            return z3.BoolVal(acc)
        else:
            vertex = path.pop(0)
            vertex_atrribute = attr.attribute_map[vertex]

            transitions: List[AutomatonTransition] = list(filter(lambda x: x.from_state == state, aut.transitions))
            ## Diverge Cases ####
            if len(transitions) == 1:
                curr = substitute(transitions[0].formula, vertex_atrribute)
                return z3.And(curr, explore_path(path, transitions[0].to_state,attr, aut, var_dict))
            elif len(transitions) == 0:
                acc = state in aut.final_states 
                return z3.BoolVal(acc)
            else:
                
                return z3.Or(
                    list(map(
                        lambda x: z3.And(substitute(x.formula, vertex_atrribute), explore_path(path, x.to_state, attr, aut, var_dict)),
                                   transitions))
                )

def query_naive_algorithm(path:List[int], attr: NodeAttributes, aut:Automaton, vars):
    solver = z3.Solver()
    f = explore_path(path, attr, vars)
    # solver.add(f)
    print(f)
    solver.add(f)
    match solver.check():
        case z3.sat:
            f = solver.model()
            print(f)
            return f 
        case _:
            return solver.check()
#######Naive Iteration Algorithm################
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
        print(transition)
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
    print(formula_vector)
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
    print(formula)
    solver = z3.Solver()
    solver.add(formula)
    match solver.check():
        case z3.sat:
            return solver.model()
        case _:
            return solver.check()




VarBound = Dict[str, float]
@dataclass
class MacroState:
    state: str 
    vars: Dict 
    para_upper_bound: VarBound 
    para_lower_bound: VarBound 


def update_macro_state(vertex_attribute,
                       attr: NodeAttributes,  
                       macro:MacroState, 
                       transition:AutomatonTransition, 
                       parameter) -> Optional[MacroState]:
                bound_vector =  z3.AstVector()
                formula = transition.formula
                varset = list(filter(
                    lambda x: x in formula, 
                    parameter.keys()
                ))
                for var in varset:
                    if var in macro.para_lower_bound:
                        variable = parameter[var]
                        upper = macro.para_upper_bound[var]
                        lower = macro.para_lower_bound[var]
                        bound_vector.push(z3.And(variable<= upper, variable>= lower))
                formula = transition.formula
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
                sat_solver.add(bound_vector)
                sat_solver.add(curr)
                print(sat_solver.assertions())

                ####check if current state is sat with the constraint ####
                match sat_solver.check():
                    case z3.unsat:
                        return None 
                    case z3.sat:
                        pass 
                
                ### To solve the upper bound 
                upper_bound_solver = z3.Optimize()
                if len(bound_vector) > 0:
                    upper_bound_solver.add(bound_vector)
                upper_bound_solver.add(curr)
                for para in varset:
                    var = parameter[para]
                    upper_bound_solver.maximize(var)
                upper_bound_solver.check()
                m = upper_bound_solver.model()
                for para in varset:
                    
                    var = parameter[para]
                    value = m.evaluate(var)
                    if isinstance(value, z3.RatNumRef):
                        val = float(m.evaluate(var).as_decimal(5))                        
                        macro.para_upper_bound[para] = val
                    else: 
                        continue
                    

                ### To solve the lower bound
                lower_bound_solver = z3.Optimize()
                if len(bound_vector) > 0:
                    lower_bound_solver.add(bound_vector)
                lower_bound_solver.add(curr)
                for para in varset:
                    var = parameter[para]
                    lower_bound_solver.minimize(var)
                lower_bound_solver.check()
                m = lower_bound_solver.model()
                for para in varset:
                    var = parameter[para]
                    value = m.evaluate(var)
                    if isinstance(value, z3.RatNumRef):
                        val = float(m.evaluate(var).as_decimal(5))
                        macro.para_lower_bound[para] = val
                    else: 
                        continue
                #MODIFY STATE 
                macro.state = transition.to_state
                return macro

def explore_with_macro_state(path:List[str],
                             attr:NodeAttributes,
                             aut:Automaton, 
                             macro_state:Optional[MacroState], 
                             parameter):
    if macro_state is None:
        return False 
    if len(path) == 0:
        return macro_state.state in aut.final_states
    else:
        vertex = path.pop(0)
        state = macro_state.state 
        vertex_atrribute = attr.attribute_map[vertex]
        transitions: List[AutomatonTransition] = list(filter(lambda x: x.from_state == state, aut.transitions))
        ## Only one successor  ####
        if len(transitions) == 1:
                new_macro_state = update_macro_state(vertex_atrribute, attr,macro_state,transitions[0], parameter)                
                return explore_with_macro_state(path, attr, aut, new_macro_state, parameter)
        ## The transition is stucked  ####
        elif len(transitions) == 0:
                acc = state in aut.final_states 
                return z3.BoolVal(acc)
        ## Multiple transitions ####
        else:
                braches = list(map(
                    lambda x: update_macro_state(vertex_atrribute, attr, macro_state, x ,parameter), 
                                   transitions))
                valid_branch = list(filter(
                    lambda x : x is not None, braches
                ))
                
                result =  list(map(
                        lambda x: explore_with_macro_state(path, attr, aut,x, parameter), 
                        valid_branch
                    ))
                return reduce(lambda x,y: x or y, result)

def query_with_macro_state(path:List[int], attr:NodeAttributes, aut:Automaton, parameter) -> bool:
  upper_bound = {}
  lower_bound = {} 
  macro_state = MacroState(
      aut.initial_state, 
      merge_dicts(attr.alphabet, parameter), 
      upper_bound,
      lower_bound      
  )
  return explore_with_macro_state(path, attr, aut, macro_state, parameter)








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


if __name__ == '__main__':
    solver = z3.Solver()
    file_path = 'example1.json'  # Path to your JSON file

    # Parse JSON file
    parsed_graph, parsed_attributes, parsed_automaton, global_vars = parse_json_file(file_path)

    # Example usage to access the parsed data
    print("Graph Database:", parsed_graph)
    print("Automaton:", parsed_automaton)
    print("Attributes: ", parsed_attributes)
    print("Alphabet: ", parsed_attributes.alphabet)
    print("Global Vars", global_vars)
    all_variables = merge_dicts(parsed_attributes.alphabet, global_vars)
    print("Product", product_graph(parsed_automaton, parsed_graph))
    print("Path", find_all_path(product_graph(parsed_automaton, parsed_graph),1,3))
    print("Query:", naive_query(parsed_automaton,parsed_graph, global_vars, 1,3, parsed_attributes) )

    # print("Formula: ", parsed_automaton.transitions[0].formula)

    # # Parse smt2 string with declared vars; returns vector of assertions, in our case always 1
    # test0 = z3.parse_smt2_string(parsed_automaton.transitions[0].formula, decls=all_variables)[0]
    # solver.add(test0)
    # print("test0: ", test0)
    # solver.check()
    # print("model 1:",solver.model())
    # test = z3.parse_smt2_string(parsed_automaton.transitions[1].formula, decls=all_variables)[0]
    # print("test:", test)
    # solver.add(test)
    # # Check model
    # solver.check()
    # print("model 2: ", solver.model())

    # # Replace age by value 2
    # test4 = (parsed_attributes.alphabet['age'])
    # test5 = (global_vars['p1'])
    # # test0[0] is the first assert in the z3 ast vector
    # expr2 = z3.substitute(test0, (test4, z3.RealVal(2.0)))
    # print("Substitute age by 2: ", expr2)
    # solver.add(expr2)
    # solver.check()

