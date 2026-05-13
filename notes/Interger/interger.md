# Evaluation PRPQ in LIA/NIA 

## Abstract Interpretation Semantics 

I follow the abstract interpretation semantics of integer sementics

### Figure 5(a): Abstract semantics for integer operations

```text
[[n]] = ceil(|n| * log2(10)) + 1
[[var]] = x
[[ite b E1 E2]] = max([[E1]], [[E2]])
[[abs E1]], [[- E1]] = [[E1]]
[[+ E1 E2]], [[- E1 E2]] = max([[E1]], [[E2]]) + 1
[[* E1 E2]] = [[E1]] + [[E2]]
[[/ E1 E2]] = max([[E1]], [[E2]])
[[mod E1 E2]] = [[E1]]
[[boolop E1 E2]] = max([[E1]], [[E2]])
```

`boolop` represents `and`, `or`, `xor`, etc.



### Reference

Benjamin Mikek and Qirun Zhang, "SMT Theory Arbitrage: Approximating Unbounded Constraints using Bounded Theories," PLDI 2024. [doi:10.1145/3656387](https://doi.org/10.1145/3656387)