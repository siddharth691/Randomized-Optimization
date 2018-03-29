import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Four peak test
#Plotting the optimal evaluation function value

with open('four_peak_test.csv') as f:
	fpt_str = f.readlines()

fpt = []
for i in range(len(fpt_str)):
	fpt.append(list(map(float, fpt_str[i].strip('\n').strip(',').split(','))))

fpt = np.array(fpt)

fig1, ax = plt.subplots()
labels = ['RHC','SA', 'GA', 'MIMIC']

for i in list(range(4)):
	ax.plot(range(9), fpt[i,:], label = labels[i])

plt.xlabel('Iterations')
plt.ylabel('Best fitness value')
plt.legend()
plt.xticks(range(9),['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Optimal fitness values with varying iterations (four peak test)')
plt.show()

#Plotting the time taken for each algorithm for varying iterations

fig2, ax = plt.subplots()

l = [4,5,6,7]
for i in l:
	ax.plot(range(9), fpt[i,:], label = labels[i-4])

plt.xlabel('Iterations')
plt.ylabel('Total time taken')
plt.legend()
plt.xticks(range(9), ['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Total time taken with varying iterations (four peak test)')
plt.show()

#Travelling salesman problem
#Plotting the optimal evaluation function

with open('travelling_salesman_test.csv') as f:
	tsp_str = f.readlines()

tsp = []
for i in range(len(tsp_str)):
	tsp.append(list(map(float, tsp_str[i].strip('\n').strip(',').split(','))))

tsp = np.array(tsp)

fig1, ax = plt.subplots()
labels = ['RHC','SA', 'GA', 'MIMIC']

for i in list(range(4)):
	ax.plot(range(9), tsp[i,:], label = labels[i])

plt.xlabel('Iterations')
plt.ylabel('Best fitness value')
plt.legend()
plt.xticks(range(9),['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Optimal fitness values with varying iterations (travelling salesman)')
plt.show()

#Plotting the time taken for each algorithm for varying iterations

fig2, ax = plt.subplots()

l = [4,5,6,7]
for i in l:
	ax.plot(range(9), fpt[i,:], label = labels[i-4])

plt.xlabel('Iterations')
plt.ylabel('Total time taken')
plt.legend()
plt.xticks(range(9), ['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Total time taken with varying iterations (travelling salesman)')
plt.show()

#Continuous peak test
#Plotting optimal function eval 
with open('cont_peak_test.csv') as f:
	cpt_str = f.readlines()

cpt = []
for i in range(len(cpt_str)):
	cpt.append(list(map(float, cpt_str[i].strip('\n').strip(',').split(','))))

cpt = np.array(cpt)

fig1, ax = plt.subplots()
labels = ['RHC','SA', 'GA', 'MIMIC']

for i in list(range(4)):
	ax.plot(range(9), cpt[i,:], label = labels[i])

plt.xlabel('Iterations')
plt.ylabel('Best fitness value')
plt.legend()
plt.xticks(range(9),['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Optimal fitness values with varying iterations (continuous peak test)')
plt.show()

#Plotting the time taken for each algorithm for varying iterations

fig2, ax = plt.subplots()

l = [4,5,6,7]
for i in l:
	ax.plot(range(9), cpt[i,:], label = labels[i-4])

plt.xlabel('Iterations')
plt.ylabel('Total time taken')
plt.legend()
plt.xticks(range(9), ['50', '100', '200', '300', '500', '700', '1000', '5000', '10000'])
plt.title('Total time taken with varying iterations (continuous peak test)')
plt.show()