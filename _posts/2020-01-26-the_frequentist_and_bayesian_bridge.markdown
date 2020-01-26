---
layout: post
title:      "The Frequentist and Bayesian Bridge"
date:       2020-01-26 17:05:25 +0000
permalink:  the_frequentist_and_bayesian_bridge
---

## Motivation for this post.
It is typical to mention at some point in a learning course on probability that statisticians divide themselves into two camps: frequentists and Bayesians. On hearing this for the first time, I recall thinking how odd it was that something could divide an entire field of study.  Of course, there are reasons that this difference exists. This post will take a look at the reasons behind the split and its practical effect on an example chosen to contrast them.

## Philosophically speaking.
At its core, the disagreement between those who identify as frequentists and those who identify as Bayesians is what is meant by the term **probability**.

* For frequentists, probabilities are related to event frequencies. 
 
If I measure the time that elapses during 9,192,631,770 cycles of the radiation produced by the transition between two levels of the cesium-133 atom repeatedly, each time I will obtain a slightly different answer. All else being equal, this difference is likely due to the statistical error inherent in the measurement device. As the number of measurements grows, the frequency of any given value will begin to indicate the probability of measuring that particular value. Frequentist probability only has meaning in a scenario of repeated measurement taking. The probability of the *true* time that elapses is meaningless. The *true* value is singluar and fixed; and, fixed values do not have a frequency distribution from which to determine their probability.

* For Bayesians, probabilities are related to event knowledge.

In this case, the probability that the *true* time that elapses during the 9.19 x 10^9 cycles of radiation from the transition between two levels of the Cs-133 atom can be found within some range has meaning. This probability is a statement on knowledge held about the measurement results. A Bayesian could claim to measure this time **T** with a probability **P(T)**. It is not required that this probability is an estimate of frequencies gathered from a large number of experiments. Instead, the concept of probability extends to the degress of certainty regarding statements that have been made. The probability **P(T)** represents the knolwedge of the value based on prior information.

Now that the difference in philosophy has been summarized, I'll next look at its practical effect on statistical analysis.


## Different approaches to measuring radiation cycles.
In the example intended to highlight the differences between frequentist and Bayesian approaches, a team is determined to measure the number of cycles of radiation produced by the transition between two levels of the cesium-133 atom in one second. For the sake of argument, we'll say this hasn't happened before and the team is genuinely curious about the result. 

The team assumes that the number of cycles is constant and exists as a fixed value, **C<sub>true</sub>**. The team performs **N** measurements where each measurement **i** has its cycles **C<sub>i</sub>** and error **e<sub>i</sub>** recorded. Errors are assumed to follow a [Gaussian](https://towardsdatascience.com/why-data-scientists-love-gaussian-6e7a7b726859) distribution typical of measurement errors. Since we are interested in the probability of a given number of events occurring in a fixed interval of time, cycle counts are assumed to follow a [Poisson](https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459) distribution.

From the set of measurements taken, what is the best parameter estimate of the true number of cycles **C<sub>true</sub>** for each approach?

For this problem,  Python and Jupyter Notebook form the environment used to construct this problem and detail its solutions.

First, generate the data:

```
# Generating cycle measurement data
import numpy as np
from scipy import stats
np.random.seed(42) 

C_true = 9192631770 / 1000000000  # true value
N = 100 
C = stats.poisson(C_true).rvs(N) # experiments
e = np.sqrt(C)  # error estimates
```

Next, preview the results:

![Cs-133 Cycles per Second](https://github.com/monstott/Blogs/raw/master/Blog2/cyclespersecond.png)

## The frequentist approach.

The maximum likelihood estimation (MLE) of a Gaussian random variable provides the likelihood of obtaining these observations for a certain mean **C<sub>true</sub>** and standard deviation **e<sub>i</sub>**. Details on formulas and their derivation can be found [here](https://www.cs.princeton.edu/courses/archive/spring08/cos424/scribe_notes/0214.pdf) and [here](https://arxiv.org/pdf/1009.2755.pdf). 

For measurements of the form **Y<sub>i</sub>** = (**C<sub>i</sub>**, **e<sub>i</sub>**), this is the probability distribution of **Y<sub>i</sub>** given **C<sub>true</sub>**:

![Maximum Likelihood Gaussian Probability Distribution](https://latex.codecogs.com/gif.latex?%24%24%20P%28Y_i%7E%7C%7EC_%7B%5Crm%20true%7D%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%20e_i%5E2%7D%7D%20%5Cexp%7B%5Cleft%5B%5Cfrac%7B-%28C_i%20-%20C_%7B%5Crm%20true%7D%29%5E2%7D%7B2%20e_i%5E2%7D%5Cright%5D%7D%20%24%24)

The likelihood function is the product of the probability distributions for each measurement in the entire set **Y**:

![Gaussian Maximum Likelihood Equation](https://latex.codecogs.com/gif.latex?%24%24%5Cmathcal%7BL%7D%28Y%7E%7C%7EC_%7B%5Crm%20true%7D%29%20%3D%20P%28Y_1%2C%20...%20%2CY_N%7E%7C%7EC_%7B%5Crm%20true%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20P%28Y_i%7E%7C%7EC_%7B%5Crm%20true%7D%29%24%24)

The log likelihood function is maximized to determine the sample estimate of the parameter since it has a convenient form and avoids the issue of numerical underflow in the presence of many potentially small numbers:

![Log Likelihood Function](https://latex.codecogs.com/gif.latex?%24%24%5Clog%5Cmathcal%7BL%7D%20%3D%20-%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cleft%5B%20%5Clog%282%5Cpi%20e_i%5E2%29%20&plus;%20%5Cfrac%7B%28C_i%20-%20C_%7B%5Crm%20true%7D%29%5E2%7D%7Be_i%5E2%7D%20%5Cright%5D%24%24)

The mean parameter estimate is found after taking the derivative of the log likelihood function with respect to **C<sub>true</sub>** and setting it equal to zero:

![Log Likelihood Derivative wrt Mean](https://latex.codecogs.com/gif.latex?%24%24%5Cfrac%7B%5Cmathrm%7Bd%7D%20%7D%7B%5Cmathrm%7Bd%7D%20F_%7B%5Crm%20true%7D%7D%5Clog%5Cmathcal%7BL%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cfrac%7BF_i%20-%20F_%7B%5Crm%20true%7D%7D%7Be_i%5E2%7D%20%3D%200%24%24)

The maximum likelihood estimate of the mean is then:

![MLE Mean](https://latex.codecogs.com/gif.latex?%24%24%20%5Chat%5Cmu%20%3D%20C_%7B%5Crm%20est%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5EN%20C_i%20/%20e_i%5E2%20%7D%7B%5Csum_%7Bi%3D1%7D%5EN%201%20/%20e_i%5E2%20%7D%20%24%24)

The [Fisher Information Matrix ](https://en.wikipedia.org/wiki/Fisher_information) provides a way to measure variation in the expected value (the MLE mean). The maximum likelihood estimate of the standard deviation is then:

![MLE Error](https://latex.codecogs.com/gif.latex?%24%24%20%5Csigma_%7B%5Crm%20est%7D%20%3D%20%5Cleft%20%28%20%5Cfrac%7B%5Cmathrm%7Bd%7D%5E2%20%5Clog%5Cmathcal%7BL%7D%20%7D%7BdC_%7Btrue%7D%5E2%7D%20%5Cright%20%29%5E%7B-1/2%7D%20%3D%20%5Cleft%28%5Csum_%7Bi%3D1%7D%5EN%20%5Cfrac%7B1%7D%7Be_i%5E2%7D%20%5Cright%29%5E%7B-1/2%7D%20%24%24)


In the case where all data points have identical errors **e<sub>i</sub>** = **e** for all measurements **i**, the mean estimate can be simplified:

![MLE Mean with Equal Errors](https://latex.codecogs.com/gif.latex?%24%24%20%5Chat%5Cmu%20%3D%20C_%7B%5Crm%20est%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%20C_i%20%24%24)

This is the arithmetic mean of the observed data.
The maximum likelihood estimate of the standard deviation also simplifies:

![Maximum Likelihood Variance Estimate](https://latex.codecogs.com/gif.latex?%24%24%20%5Csigma_%7B%5Crm%20est%7D%20%3D%20%5Cleft%20%28%20%5Cfrac%7Be%5E2%7D%7BN%7D%20%5Cright%20%29%5E%7B1/2%7D%20%24%24)

or, as is commonly seen:

![Maximum Likelihood Variance Estimate](https://latex.codecogs.com/gif.latex?%24%24%20%5Csigma_%7B%5Crm%20est%7D%20%3D%20%5Cleft%28%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cleft%28%20C_i%20-%20C_%7Best%7D%20%5Cright%20%29%5E2%20%5Cright%29%5E%7B1/2%7D%20%24%24)

This is the error estimate for the arithmetic mean in the case of Gaussian measurement errors.

Evaluate this approach for the dataset assuming the errors are not equal:

```
# frequentist approach

e2_inv = 1. / e**2
print("""
      FREQUENTIST APPROACH
      --------------------
      True Radiation Cycles - C_true = {0}
      Estimated Radiation Cycles - F_est  = {1:.2f} +/- {2:.2f}
      Number of Measurements = {3}
      """.format(C_true, 
                 (e2_inv * C).sum() / e2_inv.sum(), 
                 e2_inv.sum()**-0.5, N))
								 
>      FREQUENTIST APPROACH
>      --------------------      
>      True Radiation Cycles - C_true = 9.19263177
>      Estimated Radiation Cycles - F_est  = 7.69 +/- 0.28
>      Number of Measurements = 100
```

Using this frequentist technique, the mean estimate from the 100 measurements of the number of radation cycles (divided by 10^9) is 7.69 cycles with an error of approximately 4%. The mean parameter estimate is is off from the accepted value for radiation cycles by about 16%.

## The Bayesian approach.

The Bayesian solultion to this problem uses the probability of an event based on prior knowledge of conditions that are related to the event. This is expressed mathematically by Bayes' theorem: 

![Bayes Theorem](https://latex.codecogs.com/gif.latex?%24%24%20P%28A%7E%7C%7EB%29%20%3D%20%5Cfrac%7BP%28B%7E%7C%7EA%29%7EP%28A%29%7D%7BP%28B%29%7D%20%24%24)
 
where **A** and **B** are events and the probability of **B** is not equal to zero, **P(B) !=** 0. In the example, **A** is the event of obtaining the true radiation cycles value **C<sub>true</true>** and **B** is the event of receiving the measurements **Y**. 

The Bayes' theorem terms can be defined as follows:

* The conditional probability **P(A | B)** is the likelihood of event **A** occurring given that event **B** is true. 
In the example, this is the probability of the model parameters given the data. It is the posterior.

* The conditional probability **P(B | A)** is the likelihood of event **B** occurring given that event **A** is true.
 In the example, this is the probability of the data given the model parameters. It is the likelihood.
 
* The marginal probability **P(A)** is the probability of observing event **A**. In the example, this encodes what is known about the model prior to application of the data **Y**. It is the model prior.

* The marginal probability **P(B)** is the probability of observing event **B**. In the example, this is the probability of receiving the data as measurements.

### The prior connecting approaches.

 The prior allows inclusion of other information into conditional probability calculations. From the perspective of the prior, frequentism can be viewed as a special case of the Bayesian approach for a specific prior value. If the model prior is set so that every value has equal weighting, then **P(C<sub>true</sub>)** ∝ 1, and the prior is considered to be flat. In this case, the Bayesian probability is maximized at the same value as the frequentist approach.
 
 ### Performing the calculations.
 
Bayesian results for a one parameter problem like this one are obtained by determining the posterior probability **P(C<sub>true</sub> | Y)** as a function of **C<sub>true</sub>**. This distribution reflects our knowledge of the parameter **C<sub>true</sub>**. If the dimensions of the model grow too large, however, this direct approach becomes impossible to solve. In situations where the problem space is simply too large, Bayesian calculations depend on sampling methods. 

One class of algorithms for sampling from a probability distribution are Markov Chain Monte Carlo (MCMC) methods. Monte Carlo methods draw samples directly from the likelihood function and guess values
of the fit parameters by accepting them with the probability defined by their corresponding likelihood function value. 
Bayesian results typically reported are the entire posterior distribution over the model parameters. In order to compare values with the frequentist approach, however, the mean and standard deviation of the posterior will be presented instead.

To get an answer, first define functions for the  the posterior **P(C<sub>true</sub> | Y)**, the likelihood **P(Y | C<sub>true</sub>)**, and  the prior **P(C<sub>true</sub>)**. The data probability **P(Y)** is a normalization term and does not need to be included.

```
# define Bayesian functions

def log_prior(C_true):
    return 1  # flat prior distribution

def log_likelihood(C_true, C, e):
    return -0.5 * np.sum(np.log(2 * np.pi * e**2)
                      + (C - C_true)**2 / e**2)

def log_posterior(C_true, C, e):
    return log_prior(C_true) + log_likelihood(C_true, C, e)
```

Next, obtain the set of points:

```
# Monte Carlo Sampling

ndim = 1  # parameter space dimensions
nwalkers = 20  # walkers
nsteps = 5000  # steps
initial_guesses = 10 * np.random.rand(nwalkers, ndim)

# !pip install emcee
# import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[C, e])
sampler.run_mcmc(initial_guesses, nsteps)

sample = sampler.chain  # (nwalkers, nsteps, ndim)
sample = sampler.chain[:, :, :].ravel()  # 1 dimension
```

Lastly, visualize the outcome:

```
# histogram
plt.hist(sample, bins=100, alpha=0.8, normed=True, color='goldenrod')

# gaussian fit
C_fit = np.linspace(2, 10)
pdf = stats.norm(np.mean(sample), np.std(sample)).pdf(C_fit)
plt.plot(C_fit, pdf, '-k', lw=3)
plt.title('MCMC Sampling: Cs-133 Cycles per Second', fontsize=14)
plt.xlabel('C / 10^9', fontsize=12)
plt.ylabel('P(C)', fontsize=12)

# print results
print("""
      BAYESIAN APPROACH
      --------------------  
      True Radiation Cycles - C_true = {0}
      Estimated Radiation Cycles - C_est  = {1:.2f} +/- {2:.2f}
      Number of Measurements - N = {3}
      """.format(C_true, np.mean(sample), np.std(sample), N))
			
>     BAYESIAN APPROACH
>      --------------------  
>      True Radiation Cycles - C_true = 9.19263177
>      Estimated Radiation Cycles - C_est  = 7.68 +/- 0.31
>      Number of Measurements - N = 100
```
![MCMC Cycles per Second](https://github.com/monstott/Blogs/raw/master/Blog2/mcmc.png)

Using a Bayesian technique, the  mean estimate from the 100 measurements of the number of radation cycles (divided by 10^9) is 7.68 cycles with an error of approximately 4%. The mean parameter estimate is is off from the accepted value for radiation cycles by about 16%.

## Comparing the approaches.

The frequentist and Bayesian solutions to the problem of finding an estimate for the mean and standard deviation  of the problem resolve at nearly the same parameter estimates for the mean and standard deviation. This is no surprise, given the reduction of the Bayesian problem to a frequentist framework by the choice of prior distribution. 

Knowledge of the problem plays a large role regardelss of approach. The Gaussian distribution and identical errors are often implicitly assumed in frequentist models. If that choice was made here, however, the result would look very different:

```
# frequentist approach - equal errors

C_mean = C.sum() / N # mean estimate
C_diff = (C - C_mean)**2 # variation estimate
print("""
      FREQUENTIST APPROACH WITH EQUAL ERRORS
      --------------------------------------     
      C_true = {0}
      C_est  = {1: .2f} +/- {2:.2f} 
      N = {3} measurements
      """.format(C_true, 
                 C_mean, 
                 (C_diff.sum() / N)**0.5, N))
								 
>       FREQUENTIST APPROACH WITH EQUAL ERRORS
>      --------------------------------------     
>      C_true = 9.19263177
>      C_est  =  8.88 +/- 3.09 
>      N = 100 measurements
```

In this case, the parameter estimate of the mean is 8.88 cycles with an error of approximately 35%. This is much larger than the 4% from the approach that does not assum equal errors. Interestingly, the mean parameter estimate is is off from the accepted value for radiation cycles by only 3%, down from 16% with unequal errors. 

Regardless of the approach chosen, it is important to have an understanding of how measurements are made, how data is obtained, and how valid are the assumptions.

## Final thoughts.

This post has been an educational journey on how these two approaches are related. In the face of more complex data and models the frequentist and Bayesian approaches will undoubtedly diverge. Understanding their basis in the facts of a particular problem, however, is revealing. These two approaches require different assumptions, methods and computation choices, but managed to obtain the same result here. These approaches possess a level of harmony that I had not been able to grasp until performing this exercise. From this point of similarity, I am better positioned to identify the pros and cons of selecting a particular approach for a problem in the future. 


