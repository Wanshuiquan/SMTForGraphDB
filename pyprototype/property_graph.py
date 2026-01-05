from dataclasses import dataclass 
from typing import Set, Dict, Any, Tuple 
from itertools import product
import random 
import time 
import json

@dataclass
class PropertyGraph:
    edges: Dict[int, Tuple[Tuple[str, int]]]  # source_node ->  label, target_node)
    nodes: Set[int] # (node_id, node_type)
    attribute: Dict[int , Dict[str, Any]]  #node_id, attribute_name -> attribute_value

    def __repr__(self):
         return f"""
         edge:
            {self.edges}
         attributes:
             {self.attribute}
         """
    


def to_dict(graph:PropertyGraph): 
    return {
        "nodes": list(graph.nodes),
        "edges": graph.edges,
        "attributes":graph.attribute
    }

  

def dict_to_graph(d: Dict):
    return PropertyGraph(
        nodes=set(d["nodes"]),
        edges= {int(k): v for k, v in d["edges"].items()},
        attribute={int(k): v for k, v in d["attributes"].items()}
    )
def generate_property_graph(edge_num:int, node_num:int) -> PropertyGraph:
      """
      Docstring for generate_property_graph
      
      :param edge_num: Description
      :type edge_num: int
      :param node_num: Description
      :type node_num: int
      node attribute: since, age 
      edge label: follow, favorite, folow-anymously
      """


      nodes = set([i for i in range(node_num)])
      edges = {}
      attr = {}
      labels = ["follow", "favorite", "followanymously"]
      edge_pairs = []
      average_degree = edge_num // node_num

      for src in range(node_num):
          random.seed(time.CLOCK_MONOTONIC)

          dst = random.sample([i for i in range(node_num)], average_degree)
          for d in dst:
              edge_pairs.append((src, d))
      for src, dst in edge_pairs:
          label = random.sample(labels, 1)[0]
          if src in edges:
              edges[src].append((label, dst))
          else:
              edges[src]= [(label, dst)]

      
      for node in nodes:
          attr[node] = {}
          age_val = random.randint(15, 60)
          since_val = random.randint(1990, 2026)
          attr[node]["age"] = age_val 
          attr[node]["since"] = since_val 

      return PropertyGraph(edges, nodes, attr)


def generate_and_dump(edge_num, node_num, path):
     graph = generate_property_graph(edge_num, node_num)
     print(graph)
     with open(path, "w") as data:
          json.dump(graph, data, default=to_dict)


def load_graph(path):
     with open(path, "r") as f:
        graph = json.load(f)
        return dict_to_graph(graph)