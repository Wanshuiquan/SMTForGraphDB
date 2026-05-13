So Now we have two gaps in the submitted paper:

## Complexity Analysis of Algorithm 3 
In the paper we ignored the translation from parameteric regular expression to regular expression is exponential, the translation is a variant of the translation from a regular expression to a NFA. If we fix a parametric regular expression $r$ with $c$ formulas and $k$ variables, then we get a parametric automaton $A$ with size $2^{||r||}$. 


Let us go back to the Algorithm 3 in the paper, it is a variant of bfs on the product graph of $A$ and $G$ with size $O(2^{||r||} \cdot |G|)$. When the algorithm visits a node or an edge, it checks at most $2c$ upper and lower bounds by a linear programming solver,  $O(2c \cdot k)$ time (for linear programming), and in total we have $O(2^{||r||} \cdot |G| \cdot 2c \cdot k)$ time complexity.

This is also consistent with the original NP-Hardness proof we discussed, orinially we construct an automaton in the following way: for example, if we have $x_1 \lor \neg x_2 \lor x_3$, then 
we add three transitions between two states to represent $x_1, \neg x_2, x_3$, but if we transform the automata that encoding 3-SAT, such that two states should only have one transition, 
then the resulting automata is exponential larger then the original one. 

## Concide the NP-Hard Proof with Our Simple path semantics
now our proof of Hardness is based on a self-loop graph, we can modify the graph into a sequence graph, such that the length of the sequence graph is exact the number of clauses 
of the 3-SAT formula. 