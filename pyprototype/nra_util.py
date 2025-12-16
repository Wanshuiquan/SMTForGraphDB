from dataclasses import dataclass 
from typing import Set
import z3
EPSILON = z3.Const("epsilon", z3.RealSort())

@dataclass 
class NRA_Macro_State:
    accumulated_formula: Set[z3.Ast]
    automata_state: int  
    edge_id: int  
    node_id: int 

    def entail(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        solver.add(z3.Implies(z3.And(list(self.accumulated_formula)), other))
        res = solver.check() == z3.sat 
        solver.pop()
        return res
    

    def check_consistency(self, other: z3.Ast, solver: z3.Solver) -> bool:
        solver.push()
        solver.add(z3.And(list(self.accumulated_formula), other))
        result = solver.check() == z3.sat
        solver.pop()
        return result
    

    def visit_new_constraint(self, new_constraint: z3.Ast) -> bool:
        if self.entail(new_constraint):
            return True
        elif self.check_consistency(new_constraint):
            self.accumulated_formula.add(new_constraint)
            return True 
        else:
            return False
        

