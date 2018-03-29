import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Plotting Randomized hill climbing
with open('rhc_results.csv') as f:
	rhc_str = f.readlines()
print(rhc_str)

rhc = []
for i in range(len(rhc_str)):
	rhc.append(list(map(float, rhc_str[i].strip('\n').strip(',').split(','))))

rhc = np.array(rhc)

plt.figure()
plt.plot(range(12),rhc[0,:], range(12), rhc[2,:])
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.xticks(range(12), ['100', '200', '300', '400', '500', '700', '900', '1000', '1200', '1500', '1700', '2000'])
plt.title('Error vs iterations curve for Randomized hill climbing')
plt.legend(['Training Error','Testing Error'])
plt.show()


#Plotting simulated annaeling
cool_exp = [0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900]

sa = []
for ce in cool_exp:

	filename = 'sa_'+str(ce)+'00_results.csv'
	with open(filename) as f:
		data_str = f.readlines()

	data = []
	for i in range(len(data_str)):
		data.append(list(map(float, data_str[i].strip('\n').strip(',').split(','))))

	sa.append(np.array(data))

#Training Error
fig2, ax = plt.subplots()
for index, ce in enumerate(cool_exp):
	if(ce == 0.2):
		lw = 3
	else:
		lw = 1
	ax.plot(range(12), sa[index][0,:], label =str(ce), linewidth= lw)
plt.xticks(range(12),['100', '200', '300', '400', '500', '700', '900', '1000', '1200', '1500', '1700', '2000'])
ax.legend()
plt.ylabel('Training Error')
plt.xlabel('Iterations')
plt.title('Training Error vs iterations curve for Simulated Annealing')
plt.show()

#Testing Error
fig3, ax2 = plt.subplots()
for index, ce in enumerate(cool_exp):
	if(ce == 0.2):
		lw = 3
	else:
		lw = 1
	ax2.plot(range(12), sa[index][2,:], label =str(ce), linewidth = lw)
plt.xticks(range(12),['100', '200', '300', '400', '500', '700', '900', '1000', '1200', '1500', '1700', '2000'])
ax2.legend()
plt.ylabel('Testing Error')
plt.xlabel('Iterations')
plt.title('Testing Error vs iterations curve for Simulated Annealing')
plt.show()

#Plotting Genetic Algorithm

pop = [10, 50, 100]
mate = [5, 10, 25]
mutate = [5, 10, 25]
ga = []
for index, p in enumerate(pop):

	filename = 'ga_'+str(p)+'.000_'+str(mate[index])+'.000_'+str(mutate[index])+'.000_'+'results.csv'

	print(filename)
	with open(filename) as f:
		data_str = f.readlines()

	data = []
	for i in range(len(data_str)):
		data.append(list(map(float, data_str[i].strip('\n').strip(',').split(','))))

	ga.append(np.array(data))

#Training Error
fig4, ax3 = plt.subplots()
for index, p in enumerate(pop):
	if(index == 2):
		lw = 3
	else:
		lw = 1

	label = str(p)+','+str(mate[index])+','+str(mutate[index])
	ax3.plot(range(16), ga[index][0,:], label = label, linewidth = lw)

plt.xticks(range(16),['100', '200', '300', '400', '500', '700', '900', '1000', '1200', '1500', '1700', '2000', '4000', '8000', '16000', '20000'], rotation = 45)
ax3.legend()
plt.ylabel('Training Error')
plt.xlabel('Iterations')
plt.title('Training Error vs iterations curve for Genetic Algorithm')
plt.show()

#Testing Error
fig5, ax4 = plt.subplots()
for index, p in enumerate(pop):
	label = str(p)+','+str(mate[index])+','+str(mutate[index])

	if(index == 2):
		lw = 3
	else:
		lw = 1

	ax4.plot(range(16), ga[index][2,:], label = label, linewidth= lw)
plt.xticks(range(16),['100', '200', '300', '400', '500', '700', '900', '1000', '1200', '1500', '1700', '2000', '4000','8000','16000','20000'], rotation = 45)
ax4.legend()
plt.ylabel('Testing Error')
plt.xlabel('Iterations')
plt.title('Testing Error vs iterations curve for Genetic Algorithm')
plt.show()