README

########################################################################################################################################################
Dataset for section 1

Source: UCI Machine learning repository
Name: Covertype Data Set
Link: https://archive.ics.uci.edu/ml/datasets/Covertype
Number of instances: 581012 (Only small subset of random instances are used to perform the experiments)

########################################################################################################################################################

Instructions to Run the code:

The code is .py files and .java files. 
To run the code:

1. Install Python and java eclipse environment.
2. Clone the ABAGAIL library.
3. Follow the instructions given on this link: https://github.com/pushkar/ABAGAIL/blob/master/faq.md#user-content-how-to-use-abagail-with-eclipse
4. Run using .java files using eclipse and .py files using terminal (python filename.py)

Note: Code contains lines with reads the dataset so the dataset should be in the same location where the code is present.



########################################################################################################################################################

Folder contains:

README.txt

#Analysis
analysis.pdf (Analysis)

#Sampled dataset
cov_type_data.txt (sampled training data)
test_cov_type_data.txt (sampled test data)

#Plotting codes
plot.py (code to plot graphs for section 1)
plot1.py (code to plot graphs for section 2 with varying iterations)
plot3.py (code to plot graphs for section 2 with varying parameters)

#Code
#Section1
rhc.java (Code for neural network weights using RHC)
sa.java (Code for neural network weights using Simulated Annealing)
ga.java (Code for neural network weights using Genetic Algorithm)


#Section2
fpt.java (Four Peak Test code)
fpt_mimic.java (MIMIC varying parameters for four peak test)
tsp.java (Travelling Salesman Problem)
tsp_ga.java (GA varying parameters for Travelling Salesman Problem)
cpt.java (Continuous Peak Test)
cpt_sa.java (SA varying parameters for Continuous Peak Test)




