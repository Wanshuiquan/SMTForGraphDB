from util import Graph, NodeAttributes 
#### assume we are working with the first example 
import random, time , z3 
def generate_attribute():
    random.seed(time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    return random.random() * random.random() * 100

Hobby = ["Paint", "Reading", "Hiking", "FKK"] 
def random_path(sat = True ):
    attribute =  NodeAttributes()
    attribute.alphabet = {
        "age": z3.Real("age"), 
        "hobby": z3.String("hobby"), 
        "name": z3.String("name")
    }
    if sat: 
        attribute.attribute_map["1"] = (20, "Paint", "Li")
        attribute.attribute_map["10001"] = (30, "FKK","Li")
    else: 
        attribute.attribute_map["1"] = (20, "Paint", "Li")
        attribute.attribute_map["10001"] = (100, "FKK","Li")


    
