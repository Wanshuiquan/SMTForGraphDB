from dataclasses import dataclass 
from typing import Set, Dict, Any, Tuple 
from itertools import product
import random 
import time 
import pickle

@dataclass
class PropertyGraph:
    edges: Dict[int, Tuple[Tuple[str, int]]]  # source_node ->  label, target_node)
    nodes: Set[int] # (node_id, node_type)
    attribute: Dict[Tuple[int, str], Any]  #node_id, attribute_name -> attribute_value

def generate_property_graph(edge_num:int, node_num:int) -> PropertyGraph:
      """
      Docstring for generate_property_graph
      
      :param edge_num: Description
      :type edge_num: int
      :param node_num: Description
      :type node_num: int
      node atribute: since, age 
      edge label: follow, favorite, folowanymously
      """

      random.seed(time.CLOCK_MONOTONIC)

      nodes = set([i for i in range(node_num)])
      edges = {}
      attr = {}
      labels = ["follow", "favorite", "followanymously"]
      pairs = list(product([i for i in range(node_num)], [i for i in range(node_num)]))
      edge_pairs = random.sample(pairs, edge_num)
      for src, dst in edge_pairs:
          label = random.sample(labels, 1)[0]
          edges[(src, label)]= dst 
      
      for node in nodes:
          age_val = random.randint(15, 60)
          since_val = random.randint(1990, 2026)
          attr[(node, "age")] = age_val 
          attr[(node, "since")] = since_val 

      return PropertyGraph(edges, nodes, attr)

def generate_abnd_dump(edge_num, node_num, path):
     graph = generate_property_graph(edge_num, node_num)
     with open(path, "wb") as data:
          pickle.dump(graph, data)