import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Mimic for four peak test
#Plotting the evaluation function value

with open('fpt_mimic.csv') as f:
	fpt_str = f.readlines()

fpt = []
for i in range(len(fpt_str)):
	fpt.append(list(map(float, fpt_str[i].strip('\n').strip(',').split(','))))

fpt = np.array(fpt)
fig, ax = plt.subplots()
ax.plot(range(np.shape(fpt)[1]), fpt[0,:])
plt.xlabel("Parameters")
plt.ylabel("Optimal fitness values")
plt.title("Mimic fitness value for varying parameters")
plt.xticks(range(np.shape(fpt)[1]), ["p1","p2","p3","p4","p5","p6"])

plt.show()

#Plotting time
fig, ax = plt.subplots()
ax.plot(range(np.shape(fpt)[1]), fpt[1,:])
plt.xlabel("Parameters")
plt.ylabel("Computing time(s)")
plt.title("Mimic computing time for varying parameters")
plt.xticks(range(np.shape(fpt)[1]), ["p1","p2","p3","p4","p5","p6"])

plt.show()


#Travelling salesman problem
#Plotting the evaluation function value

with open('tsp_ga.csv') as f:
	tsp_str = f.readlines()

tsp = []
for i in range(len(tsp_str)):
	tsp.append(list(map(float, tsp_str[i].strip('\n').strip(',').split(','))))

tsp = np.array(tsp)
fig, ax = plt.subplots()
ax.plot(range(np.shape(tsp)[1]), tsp[0,:])
plt.xlabel("Parameters")
plt.ylabel("Optimal fitness values")
plt.title("GA fitness value for varying parameters")
plt.xticks(range(np.shape(tsp)[1]), ["p1","p2","p3","p4","p5"])

plt.show()

#Plotting time
fig, ax = plt.subplots()
ax.plot(range(np.shape(tsp)[1]), tsp[1,:])
plt.xlabel("Parameters")
plt.ylabel("Computing time(s)")
plt.title("GA computing time for varying parameters")
plt.xticks(range(np.shape(tsp)[1]), ["p1","p2","p3","p4","p5"])

plt.show()


#Continuous peak problem
#Plotting the evaluation function

with open('cpt_sa.csv') as f:
	cpt_str = f.readlines()

cpt = []
for i in range(len(cpt_str)):
	cpt.append(list(map(float, cpt_str[i].strip('\n').strip(',').split(','))))

cpt = np.array(cpt)
fig, ax = plt.subplots()
ax.plot(range(np.shape(cpt)[1]), cpt[0,:])
plt.xlabel("Parameters")
plt.ylabel("Optimal fitness values")
plt.title("SA fitness value for varying cooling exponent")
plt.xticks(range(np.shape(cpt)[1]), ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])

plt.show()

#Plotting time
fig, ax = plt.subplots()
ax.plot(range(np.shape(cpt)[1]), cpt[1,:])
plt.xlabel("Parameters")
plt.ylabel("Computing time(s)")
plt.title("SA computing time for varying parameters")
plt.xticks(range(np.shape(cpt)[1]), ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])

plt.show()