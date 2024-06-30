import os
from util import NodeAttributes,parse_json_file, query_with_macro_state, merge_dicts, Graph, naive_query
#### assume we are working with the first example 
import random, time , z3, networkx, time , pickle, sys 
def generate_age():
    random.seed(time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    return random.random() * 1000

Hobby = ["Paint", "Reading", "Hiking", "FKK"] 
Name = ["John", "Alice","Bob"]
def random_attribute(scale=10, sat = True ):
    attribute =  NodeAttributes()
    attribute.alphabet = {
        "age": z3.Real('age'), 
        "hobby": z3.String("hobby"), 
        "name": z3.String("name")
    }
    if sat: 
        attribute.attribute_map["0"] = (20, "Paint", "Li")
        attribute.attribute_map[f"{scale}"] = (30, "FKK","Li")
    else: 
        attribute.attribute_map["0"] = (20, "Paint", "Li")
        attribute.attribute_map[f"{scale}"] = (100, "FKK","Li")
    for i in range(1, scale):
        attribute.attribute_map[f"{i}"] = (generate_age(), random.sample(Hobby,1)[0], random.sample(Name,1)[0])
    return attribute

def random_graph(scale):
    g = Graph()
    for i in range(scale):
        g.add_node(i)
        g.add_edge(i, (i+1)%scale)
    return g
    

file_path = 'example1.json'  # Path to your JSON file
scale = 5000
attr = random_attribute(scale)
graph = random_graph(scale)  
# print(graph)
parsed_graph, parsed_attributes, parsed_automaton, global_vars = parse_json_file(file_path)
all_vars = merge_dicts(global_vars, attr.alphabet)
# print("Query:", query_with_macro_state(graph,attr,parsed_automaton,0, 4999 , global_vars) )
# print("Query:", naive_query(parsed_automaton,graph,global_vars,0, 4999 , attr) )

time_consumption = []
for i in range(1, 100):
    scale = i * 100 
    attr = random_attribute(scale)
    graph = random_graph(scale)  
    sys.stdout.write("\r" + str(i))
    sys.stdout.flush()
    t1 = time.clock_gettime(time.CLOCK_BOOTTIME)
    q1 = query_with_macro_state(graph,attr,parsed_automaton,0, scale - 1, global_vars)
    t2 = time.clock_gettime(time.CLOCK_BOOTTIME)
    q2 = naive_query(parsed_automaton,graph,global_vars,0,scale - 1, attr)
    t3 = time.clock_gettime(time.CLOCK_BOOTTIME)

    time_consumption.append((i, t2 - t1, t3-t2, q1, q2))

pickle.dump(time_consumption, open('sat.pkl'))

