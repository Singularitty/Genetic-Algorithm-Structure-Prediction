# Genetic-Algorithm-Structure-Prediction
An implementation of a genetic algorithm in Python for predicting equilibrium crystal structures for a given potential. The potential implemented here is the Daoud-Cotton model, but this can be easily changed.

Code currently isn't running due to some bugs.
Current main issues:
- Some mix up of non 1 and 0 char in the genes of the individuals
- Division by zero execption in the potential energy calculation for the given model
- Code runs slow (might need a good choice of parameters in order to be used effectively)
