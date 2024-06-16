# !/bin/env python3
import json
import z3
from typing import List, Dict
from dataclasses import dataclass 
import string 
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

@dataclass(frozen=True)
class LIA: ... 
@dataclass(frozen=True)
class STR: ...
def identify_sort(constraint):
    if "\"" in constraint:
        return STR()
    else:
        return LIA()

# explore path, state   
#         nil   state ::=  is_final_state(nil)
#        cons v::p state ::=  curr(v, state) and explore()
def explore_path(path:List[int],state, attr:NodeAttributes, aut:Automaton):
        def substitute(formulas, vertex_attribute:Dict):
                curr = z3.parse_smt2_string(formulas,decls=attr.alphabet)
                for attribute in vertex_atrribute.keys():
                    if vertex_atrribute[attribute] != None:
                        var_name = attr.alphabet(attribute)
                        val = vertex_atrribute[attribute]
                        if isinstance(val, str) or isinstance(val, string):
                           curr = z3.substitute(curr,(var_name, z3.StringVal(val)))
                        if isinstance(val, str) or isinstance(val, string):
                            curr = z3.substitute(curr,(var_name, z3.RealVal(val)))
                return curr 
        if len(path) == 0:
            acc = state in aut.final_states 
            return z3.BoolVal(acc)
        else:
            vertex = path.pop(0)
            print(attr.attribute_map)
            vertex_atrribute = attr.attribute_map[vertex]

            transitions: List[AutomatonTransition] = list(filter(lambda x: x.from_state == state, aut.transitions))
            ## Diverge Cases ####
            if len(transitions) == 1:
                curr = substitute(transitions[0].formula, vertex_atrribute)
                return z3.And(curr, explore_path(path, transitions[0].to_state,attr, aut))
            elif len(transitions) == 0:
                acc = state in aut.final_states 
                return z3.BoolVal(acc)
            else:
                braches = list(map(
                        lambda x: z3.And(substitute(x.formula, vertex_atrribute), explore_path(path, x.to_state, attr, aut)),
                                   transitions))
                return z3.Or(braches)



def query_naive_algorithm(path:List[int], attr: NodeAttributes, aut:Automaton) -> z3.Model:
    state = aut.initial_state
    formula = z3.BoolVal(True)
    solver = z3.Solver()

    f = explore_path(path, aut.initial_state,attr, aut)
    solver.add(f)
    return solver.model()
       
def query_with_macro_state(path:List[int], attr:NodeAttributes, aut:Automaton) -> bool:
    pass 
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
    print("Formula: ", parsed_automaton.transitions[0].formula)
    print("Global Vars", global_vars)
    print("Path:", explore_path(['1','2','3','1'], parsed_automaton.initial_state, parsed_attributes, parsed_automaton) )
    all_variables = merge_dicts(parsed_attributes.alphabet, global_vars)
    # Parse smt2 string with declared vars; returns vector of assertions, in our case always 1
    test0 = z3.parse_smt2_string(parsed_automaton.transitions[0].formula, decls=all_variables)[0]
    solver.add(test0)
    print("test0: ", test0)
    solver.check()
    print("model 1:",solver.model())
    test = z3.parse_smt2_string(parsed_automaton.transitions[1].formula, decls=all_variables)[0]
    print("test:", test)
    solver.add(test)
    # Check model
    solver.check()
    print("model 2: ", solver.model())

    # Replace age by value 2
    test4 = (parsed_attributes.alphabet['age'])
    test5 = (global_vars['p1'])
    # test0[0] is the first assert in the z3 ast vector
    expr2 = z3.substitute(test0, (test4, z3.RealVal(2.0)))
    print("Substitute age by 2: ", expr2)

