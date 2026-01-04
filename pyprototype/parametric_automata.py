from dataclasses import dataclass 
from typing import Set, Dict, Any, Tuple 
import json 

@dataclass 
class ParametricAutomaton:
    initial_state: int
    accepting_states: Set[int]
    transitions: Dict[int, Dict[int, Tuple[str, str]]]  # source_state -> ((edge_label, constraint), target_state)
    global_parameters: Set[str]

    def __repr__(self):
        return f"""
        Init:{self.initial_state},
        Final States:{self.accepting_states},
        Transitions: {self.transitions}
"""

def read_json(path):
    with open(path) as f:
        dic = json.load(f)
        init = dic["Init"]
        global_var = set(dic["Global Variables"])
        accept = set(dic["Final States"])
        transition = {}
        for ele in dic["Transitions"]:
           src = ele["from"]
           dst = ele["to"]
           label =  ele["label"]
           constraint = ele["formula"]
           if src in transition:
              transition[src][dst] = (label, constraint)
           else:
              transition[src] = {}
              transition[src][dst] = (label, constraint)
        return ParametricAutomaton(init, accept, transition, global_var) 


