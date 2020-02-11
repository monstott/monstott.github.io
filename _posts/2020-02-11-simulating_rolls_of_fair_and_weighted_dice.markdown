---
layout: post
title:      "Simulating Rolls of Fair and Weighted Dice"
date:       2020-02-11 23:59:37 +0000
permalink:  simulating_rolls_of_fair_and_weighted_dice
---


## Objective.
The goal of this article is to display knowledge of basic statistics and Python programming. 
These skills are required for a data-focused career and are often a part of technical interviews and screenings.

## Problem.
The task is to write a program that simulates and outputs the result of 100 rolls of a 20-sided die. 

## Questions.
 
**1.** Estimate the average and standard deviation of the rolls for a fair die. 

**2.** Estimate the average and standard deviation of the rolls for a weighted die.
 
**3.** Estimate the expected value and standard deviation of a single roll of a fair die.

**4.** Estimate the expected value and standard deviation of a single roll of a weighted die.

 
## Solutions.

### Solution for Question 1.
- I've created a function that takes inputs of the number of trials and the number of sides on the fair die.
- The function simulates random rolls of the die. 
- Each roll is added to a list meant to record each cast of the die. 
- Once the experiment is complete, the average and standard deviation are computed from this list.
- Each roll also increments the value of a dictionary key for the resulting die side. 
- Once the experiment is complete, the dictionary is used to visualize the results.

```
# Import 
import matplotlib.pyplot as plt
import numpy as np
import random

# Simulation
def simulateDie(n=100, s=20):
    '''
    FUNCTION: Simulate `n` rolls of a fair `s`-sided die.
    '''
    results = dict.fromkeys(range(1, s+1), 0) # initialized dictionary for results
    store = []
    for i in range(n): # number of rolls
        roll = random.randint(1, s) # random roll
        results[roll] += 1
        store.append(roll)    
    avg = np.mean(store)
    stdev = np.std(store)
    
    return results, avg, stdev  

# Experiment
trials = 100 # number of rolls in experiment
sides = 20 # number of sides on die
results, avg, stdev = die_simulation(trials, sides) #simulation
print('Estimates for 100 Rolls of a Fair 20-Sided Die')
print('-'*45)
print('Average:', round(avg, 2))
print('Standard Deviation:', round(stdev, 2))
 
# Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.bar(range(len(results)), list(results.values()), align='center', color='cornflowerblue')
plt.xticks(range(len(results)), list(results.keys()), fontsize=14)
ax.set_title('Simulation: 100 Rolls of a Fair 20-Sided Die', fontsize=20)
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Side', fontsize=14)
plt.show();

> Estimates for 100 Rolls of a Fair 20-Sided Die
> ---------------------------------------------
> Average: 9.66
> Standard Deviation: 5.64
```

![Fair Die](https://github.com/monstott/Blogs/raw/master/Blog8/fairdie.png)

### Solution for Question 2.
- I've created a function that takes inputs of the number of sides on a die, the weighted side, and the percentage of outcomes that the weighted number is anticipated to be cast.
- The function creates a list representing the weights of each side on the die. 
- The simulation is then run. The weights list is used as an index to determine the roll. 
- Each roll is added to a list meant to record each cast of the die. 
- Once the experiment is complete, the average and standard deviation are computed from this list.
- This list is also used to visualize the results.

```
# Import
from itertools import repeat 

# Weights
s = 20 
def weightDie(s, num, pct):
    '''
    FUNCTION: Assign weights to a weighted die, 
    where side `num` in die with sides `s` is rolled a percentage `pct` of the time.
    '''
    weights = []
    for i in range(1, s+1): 
        if i == num:
            weights.extend(repeat(i, int((2*(s - 1) * pct))))
        else: 
            weights.append(i)
            
    return weights

# Calculate Weights
weights = weightDie(s, 1, 0.5) # weight the outcome 1 as half of all rolls

# Simulation
n = 100
w_store = []
w_results = [0]*s  
for i in range(n):
    idx = random.randint(0, len(weights)-1)
    w_roll = weights[idx]
    w_results[w_roll - 1] += 1  
    w_store.append(w_roll)    
w_avg = np.mean(w_store)
w_stdev = np.std(w_store)
    
# Results 
print('Estimates for 100 Rolls of a Weighted 6-Sided Die')
print('(where a roll of 1 occurs half of the time)')
print('-'*45)
print('Average:', round(w_avg, 2))
print('Standard Deviation:', round(w_stdev, 2))   
    
# Plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.bar(range(len(w_results)), w_results, align='center', color='skyblue')
plt.xticks(range(len(w_results)), range(1, len(w_results)+1), fontsize=14)
ax.set_title('Simulation: 100 Rolls of a Weighted 6-Sided Die', fontsize=20)
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Side', fontsize=14)
plt.show();

> Estimates for 100 Rolls of a Weighted 6-Sided Die
> (where a roll of 1 occurs half of the time)
> ---------------------------------------------
> Average: 5.96
> Standard Deviation: 6.55
```

![Weighted Die](https://github.com/monstott/Blogs/raw/master/Blog8/weightdie.png)

### Solution for Question 3.
- The equal probability (1/20) of each face of the fair die is multiplied with each outcome and then summed to compute the expected outcome.
- The standard deviation is the average deviation of each outcome from the expected outcome.

```
# For a random variable: [Mean Estimate] = Sum( P(1) * 1 + ... + P(20) * 20)
# For a fair 20-sided die: P(n) = 1 / 20, where n is any number 
avg_1 = sum((1/s)*x for x in range(1, s+1))
stdev_1 = np.std(range(1, s+1))
print('Estimates for 1 Rolls of a Fair 20-Sided Die')
print('-'*45)
print('Expected Value:', round(avg_1, 2))
print('Standard Deviation:', round(stdev_1, 2))

> Estimates for 1 Rolls of a Fair 20-Sided Die
> ---------------------------------------------
> Expected Value: 10.5
> Standard Deviation: 5.77
```

### Solution for Question 4. 
- The probability of each face of the fair die is calculated from the side weights and is multiplied with each outcome and then summed to compute the expected outcome.
- The standard deviation is the average deviation of each outcome from the expected outcome.

```
# For a random variable: [Mean Estimate] = Sum( P(1) * 1 + ... + P(20) * 20)
# For a weighted 20-sided die the probabilities must be calculated differently
from collections import Counter

w_counts = Counter(weights)
w_avg_1 = sum((w_counts[x]/len(weights))*x for x in range(1, s+1))

sd_tot = 0
for i in range(1, s+1):
    sd_tot += (i - w_avg_1)**2
    w_stdev_1 = np.sqrt(sd_tot / s)
w_stdev_1

print('Estimates for 1 Rolls of a Weighted 20-Sided Die')
print('(where a roll of 1 occurs half of the time)')
print('-'*45)
print('Expected Value:', round(w_avg_1, 2))
print('Standard Deviation:', round(w_stdev_1, 2))

> Estimates for 1 Rolls of a Weighted 20-Sided Die
> (where a roll of 1 occurs half of the time)
> ---------------------------------------------
> Expected Value: 6.0
> Standard Deviation: 7.31
```

## Final Thoughts.

The 100 roll and single roll estimates are not the same because they are providing summary statistics for different activities. The fair and weighted die estimates are not the same because the probability of each outcome is no longer equal when weights are applied, affecting results. 

The 100 roll estimate requires an experiment to be performed with the 20-sided die. In this experiment 100 rolls are cast and the results are recorded. The questions here are (1.) what is the average roll in the data? and (2.) how does the data vary from the average roll? The probability distribution of the die outcomes that is made by the simulation is used to answer these questions. For the weighted die, the side set to appear half of the time (1) skews the statistics.

The second question does not require an experiment to be performed. It requires knowledge of how expected values are calculated. The expected value assumes that the probabilities for each outcome are equal (1/20) for the fair die. For the weighted die, the probabilities of each outcome must be determined from the weights. To calculate the expected value, the probability value is multiplied by each possible outcome and the results are summed. The standard deviation provides a measurement of how far each outcome is from the expected value. It is calculated by taking the square root of the average deviation of each outcome from the expected value.


