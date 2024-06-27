from util import Graph, NodeAttributes 
#### assume we are working with the first example 
import random, time , z3 
def generate_age():
    random.seed(time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    return random.random() * random.random() * 100

Hobby = ["Paint", "Reading", "Hiking", "FKK"] 
Name = ["John", "Alice","Bob"]
def random_attributr(sat = True ):
    attribute =  NodeAttributes()
    attribute.alphabet = {
        "age": z3.Real("age"), 
        "hobby": z3.String("hobby"), 
        "name": z3.String("name")
    }
    if sat: 
        attribute.attribute_map["1"] = (20, "Paint", "Li")
        attribute.attribute_map["100001"] = (30, "FKK","Li")
    else: 
        attribute.attribute_map["1"] = (20, "Paint", "Li")
        attribute.attribute_map["100001"] = (100, "FKK","Li")
    for i in range(2, 100001):
        attribute.attribute_map[f"{i}"] = (generate_age(), random.sample(Hobby,1)[0], random.sample(Name,1)[0])
    return attribute

def random_path():
    pass
    


    
