# Mutlimodality

This Python package estimates whether an empirical two-dimensional distribution 
is unimodal. It computes a unimodality index. Here is an examlpe of use:

```python
results = mm.multimodality_analysis(x, y)
ua = mm.unimodality_analysis(results)
print(ua["unimodality_index"])
```
