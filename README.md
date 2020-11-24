# Genetic-Algorithm-Structure-Prediction
An implementation of a genetic algorithm in Python for predicting equilibrium crystal structures for a given potential. 

# Features
- Predicts ground-state (GS) structures in 2D for a given potential;
- Change parameters in the program easily by acessing and editing the settings.py file;
- Simple change of the interaction governing potential by selecting the desired one in the settings.py file (currently only has daoud-cotton model implemented);
- Easy implementation of other potentials by adding their own files to the program folder.

Currently includes two versions of the script, one GA.py which only runs in serial and a paralellized version GA_Parallel.py which has some functions running the parellel to speed up the computations.

# To do
- Some more testing
- Implement a potential for elastic multipole interactions
- Do some clean up of the code
- Attempt to paralellize more of the code to improve performance
