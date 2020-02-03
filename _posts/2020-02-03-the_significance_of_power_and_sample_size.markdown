---
layout: post
title:      "The Significance of Power and Sample Size"
date:       2020-02-03 07:18:10 +0000
permalink:  the_significance_of_power_and_sample_size
---

### Motivation.
Power analysis is a powerful tool capable of informing the design of experiments and analysis of their results. The power of an experiment reflects the confidence associated with the conclusions reached from its results. As a result, power is an indispensible consideration for successful statistical hypothesis testing. Another fundamental criterion of hypothesis testing is sample size. Determining the optimal sample size is very important for organizations. It directly relates to business investments of time, money, and employee resources; furthermore, it is capable of dictating whether results are considered statistically significant or not. This post will examine the effects of power and sample size in the statistical inference process.

### Statistical Hypothesis Testing.
**Statistical Hypothesis Tests** make an assumption about the outcome of an experiment. The assumed outcome is the **Null Hypothesis**, denoted as *H<sub>0</sub>*. The null hypothesis holds the default position that no significant effect is present between two measured phenomena (e.g., samples means are the same). The outcome tested against the null hypothesis is the **Alternative Hypothesis**, denoted as *H<sub>a</sub>*. The alternative hypothesis argues that a statistically significant effect exists (e.g., sample means are different).

Statistical hypothesis tests are interpreted using a p-value. The **p-value** is the probability of obtaining the result, or a more extreme result, given the data and that the null hypothesis is true.

In order to interpret the p-value of a test the significance level must be specified. The **Significance Level** is the threshold value  that sets the boundary for determining if the test is statistically significant. The result of a significance test is statistically significant if the p-value is less than the significance level. If the p-value is less than the significance level then the null hypothesis is rejected. A common value used for the significance level is 5%. The significance level is denoted by *α*. 

**Relationship between p-Value and Significance Level:**

* *p > α*: Fail to Reject H<sub>0</sub> 
*  *p ≤ α*: Reject H<sub>0</sub> 

Since the p-value is a probability, the statistical test result could be wrong and the result may be different in truth. There are two types of errors that can be made from the interpretation of statistical hypothesis tests.

**Interpretation Errors:**
* *Type I*: Reject the H<sub>0</sub> when there is no significant effect.
 * For Type I errors, the p-value is optimistically small and the effect identified by the test is a *False Positive*.
* *Type II*: Fail to Reject H<sub>0</sub> when there is a significant effect.
 *  For Type II errors, the p-value is pessimistically large and the absence of an effect identified by the test is a *False Negative*.
   
Since the significance level sets the boundary for rejecting the null hypothesis, it can be interpreted as the probability of rejecting the null hypothesis given that it is true. In other words, this is the probability of making a Type I error (false positive).

The plot below illustrates the relationship between the null hypothesis, alternative hypothesis, type I error,  and type II error.

```
# type I and II error plot
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
# left type I 
plt.fill_between(x=np.arange(-4, -2, 0.01), y1=stats.norm.pdf(np.arange(-4,-2,0.01)), facecolor='red', alpha=0.4)
# null hypothesis
plt.fill_between(x=np.arange(-2,2,0.01), y1=stats.norm.pdf(np.arange(-2,2,0.01)), facecolor='gray', alpha=0.4)
# right type I
plt.fill_between(x=np.arange(2,4,0.01), y1=stats.norm.pdf(np.arange(2,4,0.01)), facecolor='red', alpha=0.6)
# left alternative hypothesis
plt.fill_between(x=np.arange(-4,-2,0.01), y1=stats.norm.pdf(np.arange(-4,-2,0.01),loc=3, scale=2), facecolor='gray', alpha=0.4)
# middle alternative hypothesis
plt.fill_between(x=np.arange(-2,2,0.01), y1=stats.norm.pdf(np.arange(-2,2,0.01),loc=3, scale=2), facecolor='blue', alpha=0.4)
# right alternative hypothesis
plt.fill_between(x=np.arange(2,10,0.01), y1=stats.norm.pdf(np.arange(2,10,0.01),loc=3, scale=2), facecolor='gray', alpha=0.4)
# format
plt.xticks([]) 
plt.yticks([]) 
# text
plt.text(x=-1, y=0.2, s='Null Hypothesis', fontsize=14)
plt.text(x=2.5, y=0.15, s='Alternative', fontsize=14)
plt.text(x=2.2, y=0.01, s='Type I Error', fontsize=14)
plt.text(x=-3.4, y=0.01, s='Type I Error', fontsize=14)
plt.text(x=0, y=0.03, s='Type 2 Error', fontsize=14);
```

![error types plot](https://github.com/monstott/Blogs/raw/master/Blog7/errortypes.png)

Type I errors are shaded red and type II errors are shaded blue in this exaggerated view. The significance level shown is α = 0.05.
### Power Analysis.

**Statistical Power** is the probability that the statistical test correctly rejects the null hypothesis. In other words, it is the probability of a positive result given that the null hypothesis is false (i.e., the alternative hypothesis is true). As the statistical power for an experiment increases, the probability of making a Type II error (false negative) decreases. Power can be expressed as the inverse of the probability of a Type II error.

**Relationship between Power and Type II Error:**
* Power = 1 - β, where P(Type II Error) = β.
* Power = P(Reject H<sub>0</sub> | H<sub>a</sub> is True) = 1 - P(Fail to Reject H<sub>0</sub> | H<sub>0</sub> is False).

As a result of the relationship between power and error, experimental results with very low statistical power are likely to reach incorrect conclusions from the results obtained. To mitigate this risk, a minimum level of statistical power is set. It is common for experiments to be designed with a statistical power of at least 80%. This translates to a 20% probability of a Type II error. 

**Effect Size**  quanitifies the magnitude of a phenomenon in the population. Effect sizes are used to complement the results from statistical hypothesis tests. A hypothesis test quantifies the likelihood of observing the data given the absence of any effect. Contrastingly, the effect size quantifies the size of the effect assuming the effect is present. Ideally, the product of an experiment is one (or more) effect size measures that compliment the p-value. This approach prevents the acceptance of statistically significant but trivial results.

Effect sizes methods are organized into groups based on the type of effect quantified. 

**Effect Size Types:**
* *Association*: methods that quantify a relationship between variables.
 * As an example, Pearson's Correlation Coefficient *r* measures the linear association between two numeric variables.
 * This measure is unitless and can be interpreted in a standard way; values range from a negative perfect relationship (r = -1), through no relationship (r = 0), and up to a positive perfect relationship (r = 1).
* *Difference*: methods that quantify the difference between variables.
 * As an example, Cohen's *d* measures the difference between the means of two variables with Gaussian distributions.
 * This measure has standardized units that describe the difference with standard deviations; standard values are often used to refer to a small effect size (d = 0.20), medium effect size (d = 0.50), and large effect size (0.80).

**Components of Power Analysis:**

* *Sample Size*: the number of observations in the sample.
* *Effect Size*: the magnitude of a result in the population.
* *Significance Level*: the probability of rejecting the null hypothesis if it is true.
* *Statistical Power*: the probability of accepting the alternative hypothesis if it is true.

All of the components of power analysis are related. **Power Analysis** is the estimation of one of these four parameters given values for the other three. As an example, the minimum sample size required for an experiment can be estimated from the effect size, significance level, and statistical power. This is possible by setting practical values for the three parameters and calculating the sample size estimate. By adjusting values and recalculating, curves of one parameter against another can be developed to inform the experimental design. Before conducting a study, however, the significance level should be specified so as not to negatively influence the interpretation of results.

### Application to the Student’s t-Test.

The Student’s t-Test is a statistical hypothesis test that compares the means from two samples of Gaussian variables. This test assumes the two samples have equal sizes and variance. The null hypothesis of this test is that the sample populations have the same mean (the samples are from the same population). The alternative hypothesis is that the sample populations have different means (the samples are from different populations). 
The significance level for this example will be set at the common value of 5% (α = 0.05). The statistical power wil be the common value of 80% (1 - β = 0.80). The effect size measure for the test is the standardized difference of group means (Cohen's d). 

**Cohen's d Measure:**
 * d = ( μ<sub>1</sub> - μ<sub>2</sub> ) / s, where μ<sub>1</sub> is the mean of the first sample, μ<sub>2</sub> is the mean of the second sample, and s is the pooled standard deviation of both samples.
  * s = sqrt( ( ( n<sub>1</sub> - 1 ) s<sub>1</sub><sup>2</sup> + ( n<sub>2</sub> - 1 ) s<sub>2</sub><sup>2</sup> ) / ( n<sub>1</sub> + n<sub>2</sub> - 2 ) ), where n<sub>1</sub> is first sample size, n<sub>2</sub> is the second sample size, s<sub>1</sub><sup>2</sup> is the first sample variance, and s<sub>2</sub><sup>2</sup> is the second sample variance.

The effect size will be set at a medium level (Cohen's d = 0.50). Together, these three arguments will be used to constrain the estimate of the sample size. In Python, the statsmodels method used to solve for one parameter of the power of a two sample t-test is `solve_power()` from the `TTestIndPower` class.

```
# power analysis: sample size estimation
from statsmodels.stats.power import TTestIndPower

alpha = 0.05
power = 0.8
es = 0.50

solver = TTestIndPower()
size = solver.solve_power(nobs1=None, alpha=alpha, power=power, effect_size=es, ratio=1.0)

print('Sample Size: %.2f' % size)
print('Effect Size (standard deviations): %.2f' % es)
print('Prob. of detecting a true effect: %.f%%' % (power*100))
print('Prob. of failing to detect a true effect: %.f%%' % ((1-power)*100))
print('Prob. detecting a false effect: %.f%%' % (alpha*100))

> Sample Size: 63.77
> Effect Size (standard deviations): 0.50
> Prob. of detecting a true effect: 80%
> Prob. of failing to detect a true effect: 20%
> Prob. detecting a false effect: 5%
```

The estimate for the minimum number of observations required in each sample to detect a medium-sized effect with error rates of 5% for Type I and 20% for Type II  is 64.

Power curves visualize how changes in sample size impact statistical power values for different effect sizes at a given significance level. I'll plot curves for small (0.20), medium (0.50), and large (0.80) effect sizes for 3 significance levels, α  = [0.01, 0.05, 0.10], with the `plot_power()` method.

```
# power curves: sample size by effect size for alpha = 0.05
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

es = np.array([0.2, 0.5, 0.8])
sizes = np.array(range(5, 100))
curves = TTestIndPower()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

curves.plot_power(dep_var='nobs', nobs=sizes, effect_size=es, alpha=0.01, ax=ax1)
ax1.set_title('Power vs Sample Size by Effect Size - alpha = 0.01', fontsize=18)
ax1.set_ylabel('Power', fontsize=14)
ax1.set_xlabel('')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(loc='best', prop={'size':14})

curves.plot_power(dep_var='nobs', nobs=sizes, effect_size=es, alpha=0.05, ax=ax2)
ax2.set_title('Power vs Sample Size by Effect Size - alpha = 0.05', fontsize=18)
ax2.set_ylabel('Power', fontsize=14)
ax2.set_xlabel('')
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.legend(loc='best', prop={'size':14})

curves.plot_power(dep_var='nobs', nobs=sizes, effect_size=es, alpha=0.10, ax=ax3)
ax3.set_title('Power vs Sample Size by Effect Size - alpha = 0.10', fontsize=18)
ax3.set_ylabel('Power', fontsize=14)
ax3.set_xlabel('Sample Size', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.legend(loc='best', prop={'size':14})

plt.show()
```

![2-sample power curves](https://github.com/monstott/Blogs/raw/master/Blog7/2samp_powercurves.png)

**Observations:**
* As the sample size increases, the statistical power value also increases until the curve reaches maximum power or is stopped by the axes limits. 
* At all sample sizes, the large effect size has greater power than the medium effect size, which has greater power than the small effect size. 
* A greater significance level will have higher power than a lesser significance level at a given point.


Power planes expand the visualization of parameter relationships to three dimensions. This can be performed with the `plot_surface()` method of the `Axes3D` class. I'll examine the effect on statistical power across combinations of effect size and sample size with the statistical level set at α = 0.05.

```
# surface of power, sample size, and effect size for alpha = 0.05
from mpl_toolkits.mplot3d import Axes3D

# power finding function
@np.vectorize
def find_power(ss, es):
    power = TTestIndPower().solve_power(nobs1=ss, effect_size=es, alpha=0.05)
    
    return power
# values
X, Y = np.meshgrid(np.linspace(10, 1000, 100), np.linspace(0.01, 1, 50))
X = X.T
Y = Y.T
Z = find_power(X, Y) 
# plot
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot_surface(ax, X, Y, Z, cmap='winter')
ax.set_title('Surface of Power, Sample and Effect Sizes (alpha = 0.05)', fontsize=18)
ax.set_xlabel('Sample Size', fontsize=14)
ax.set_ylabel('Effect Size', fontsize=14)
ax.set_zlabel('Power', fontsize=14)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
```

![2-sample power plane](https://github.com/monstott/Blogs/raw/master/Blog7/powerplane.png)

**Observations:**
* The surface plot confirms the understanding that statistical power increases with effect size and sample size. 
* This view makes it possible to judge the relative impact of each parameter on power values. 
* Medium and large effect sizes (above 0.5) have high power values (above 0.80) across nearly the entire range of sample sizes.
* High sample sizes (above 200) do not translate to high power values for low effect sizes (0.20). 

### Application to the Two Proportion Z-test 
The two proportion Z-test determines whether a population proportion p<sub>1</sub> is equal to another population proportion p<sub>2</sub>. The null hypothesis is that there is no difference between the proportions ( p<sub>1</sub> -  p<sub>2</sub> = 0). The alternative hypothesis is that there is a difference between the proportions  (p<sub>1</sub> -  p<sub>2</sub> ≠ 0). The **Test Statistic** measures the degree of agreement between the samples and the null hypothesis. The statistic for this test is denoted by *Z*. The number of standard deviations that the sample difference is away from the null hypothesis can be interpreted from the value of Z. If the test statistic is outside the quantiles of the standard normal distribution that correspond to the significance level and degrees of freedom then the null hypothesis is rejected. As an extension of this relationship with the null hypothesis, the p-value is the probability of obtaining the test statistic.

**Z-Test Statistic:**
* Z = ( p<sub>1</sub> - p<sub>2</sub> ) / sqrt( p̂ (1 - p̂ )( 1 / n<sub>1</sub> + 1 / n<sub>2</sub> ) ), where p̂ = ( n<sub>1</sub> **·** p<sub>1</sub> + n<sub>2</sub> **·** p<sub>2</sub> ) / ( n<sub>1</sub> + n<sub>2</sub> )
 * The pooled sample proportion p̂ is the proportion of positive results in the two samples combined.

**Components of Sample Size Analysis:**
* *Sample Size*: the number of observations in the sample.
* *Significance Level*: the probability of rejecting the null hypothesis if it is true.
* *Effect Size*: the difference to be detected (p<sub>1</sub> - p<sub>2</sub>).
* *Proportion Values*:  the values of the sample proportions used in the test (p<sub>1</sub> and p<sub>2</sub>).
 * As an example, the sample size required to detect a difference between sample proportions of 20% and 25% is different than the sample size required for sample proportions of 70% and 75%. 

Sample size can be found from the significance level, the difference between p<sub>1</sub> and p<sub>2</sub>, and the absolute value of p<sub>1</sub>. The sample proportion p<sub>2</sub> can be calculated from the difference and p<sub>1</sub>. 

The first step in determining sample size is to define a function `prop_z()` that returns the test statistic Z.

```
# 2 proportion Z-test statistic
def prop_z(p1, p2, n1, n2):
    pool_p = (n1*p1 + n2*p2) / (n1 + n2)
    
    return (p2 - p1) / math.sqrt(pool_p * (1 - pool_p) * ((1. / n1) + (1. / n2)))
```

The next step is to define a function `find_size()` that returns the sample size that produces a statistically significant result (p-value < α) from the proportion values, their difference, and the significance level. Since the distribution of the test statistic Z is approximately normal, the p-values is obtained from interrogating how unlikely the test statistic is within the standard normal distribution given the null hypothesis. The sizes of both samples are assumed to be equal in order to simplify this process. 

```
# find the sample size requirement from proportions and significance level
def find_size(p1, p_diff, alpha):
    n = 1
    while True:
        z = prop_z(p1, p1+p_diff, n1=n, n2=n) # assume equal sizes
        p = 1 - stats.norm.cdf(z)
        if p < alpha:
            break
        n += 1
        
    return n
```

The last step is to visualize the results. The sample proportion difference is set at p<sub>1</sub> -  p<sub>2</sub> = 0.05, and the significance levels reviewed are α = [0.01, 0.05, 010].

```
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# sample sizes for changes in p1 - 5% difference
p_diff = 0.05
p1_list = [x/100 for x in range(100-int(p_diff*100)+1)]

output = []
for p1 in p1_list:
    alpha1 = 0.01
    alpha2 = 0.05
    alpha3 = 0.10
    
    item1 = {}
    item1['Difference'] = p_diff
    item1['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha1)
    item1['p1 Percentage'] = int(p1 * 100)
    item1['alpha'] = alpha1    
    output.append(item1)   
    
    item2 = {}
    item2['Difference'] = p_diff
    item2['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha2)
    item2['p1 Percentage'] = int(p1 * 100)
    item2['alpha'] = alpha2    
    output.append(item2)   
    
    item3 = {}
    item3['Difference'] = p_diff
    item3['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha3)
    item3['p1 Percentage'] = int(p1 * 100)
    item3['alpha'] = alpha3    
    output.append(item3) 

df = pd.DataFrame(output)

# plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style('ticks')
sns.pointplot(x='p1 Percentage', y='Sample Size', hue='alpha', ax=ax, data=df)
ax.set_title('5% Difference: Sample Size vs p1 Proportion (by Significance Level)', fontsize=18)
ax.set_xlabel('p1 Percentage (%)', fontsize=14)
ax.set_ylabel('Sample Size', fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.show()
```

![sample size by p1 proportions](https://github.com/monstott/Blogs/raw/master/Blog7/p1Diff.png)

**Observations:**
* for all significance levels, the minimum sample size required to prove a 5% effect is greatest at a p<sub>1</sub> proportion percentage of 50%. 
* As the significance level increases, the sample size required to detect a desired effect decreases. 
* The curves are symmetrical which makes sense given the trade between p<sub>1</sub> and  p<sub>2</sub> that occurs when moving across the x-axis.

I'll now compare this with a higher sample proportion difference,  p<sub>1</sub> -  p<sub>2</sub> = 0.10. 

```
# sample sizes for changes in p1 - 10% difference
p_diff = 0.10
p1_list = [x/100 for x in range(100-int(p_diff*100)+1)]

output = []
for p1 in p1_list:
    alpha1 = 0.01
    alpha2 = 0.05
    alpha3 = 0.10
    
    item1 = {}
    item1['Difference'] = p_diff
    item1['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha1)
    item1['p1 Percentage'] = int(p1 * 100)
    item1['alpha'] = alpha1    
    output.append(item1)   
    
    item2 = {}
    item2['Difference'] = p_diff
    item2['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha2)
    item2['p1 Percentage'] = int(p1 * 100)
    item2['alpha'] = alpha2    
    output.append(item2)   
    
    item3 = {}
    item3['Difference'] = p_diff
    item3['Sample Size'] = find_size(p1=p1, p_diff=p_diff, alpha=alpha3)
    item3['p1 Percentage'] = int(p1 * 100)
    item3['alpha'] = alpha3    
    output.append(item3) 

df = pd.DataFrame(output)

# plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style('ticks')
sns.pointplot(x='p1 Percentage', y='Sample Size', hue='alpha', ax=ax, data=df)
ax.set_title('10% Difference: Sample Size vs p1 Proportion (by Significance Level)', fontsize=18)
ax.set_xlabel('p1 Percentage (%)', fontsize=14)
ax.set_ylabel('Sample Size', fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.show()
```

![sample size by p1 proportions 2](https://github.com/monstott/Blogs/raw/master/Blog7/p1Diff2.png)

**Observations:**
* The significance level curves maintain their shapes moving from the last plot with a 5% difference to this plot with a 10% difference.
* For this increased difference, the required sample sizes to find the desired effect are reduced greatly. Values are one-fourth of the size, with the maximum decreasing from 1,000 to 250.

Since the sample size required to discern a difference between p<sub>1</sub> and p<sub>2</sub> depends on the absolute value of p<sub>1</sub>, there exists a value of p<sub>1</sub> that requires the largest sample size. This p<sub>1</sub> value represents an upper boundary on the number of samples to collect. In practice, when the proportions are unknown, using the highest required sample size is a simple way to guarantee the actual proportion requirement is satisfied.

To find the sample size required for this boundary case, I'll set the first proportion at the maximum, p<sub>1</sub> = 0.50. The sample size will be a function of the proportion difference, which I'll range from 1% to 15%. Additionally, curves will be plotted for the same significance levels that have been used througout this investigation. 

```
# maximum sample size required - p1 = 50%
p_diffs = [x for x in range(1, 16)]

output = []
for p_diff in p_diffs:
    alpha1 = 0.01
    alpha2 = 0.05
    alpha3 = 0.10
    
    item1 = {}
    item1['Difference'] = p_diff
    item1['Sample Size'] = find_size(p1=0.5, p_diff=p_diff/100, alpha=alpha1)
    item1['p1 Percentage'] = '50%'
    item1['alpha'] = alpha1    
    output.append(item1)
    
    item2 = {}
    item2['Difference'] = p_diff 
    item2['Sample Size'] = find_size(p1=0.5, p_diff=p_diff/100, alpha=alpha2)
    item2['p1 Percentage'] = '50%'
    item2['alpha'] = alpha2    
    output.append(item2)
    
    item3 = {}
    item3['Difference'] = p_diff 
    item3['Sample Size'] = find_size(p1=0.5, p_diff=p_diff/100, alpha=alpha3)
    item3['p1 Percentage'] = '50%'
    item3['alpha'] = alpha3    
    output.append(item3)

df = pd.DataFrame(output)

fig, ax = pyplot.subplots(figsize=(12, 8))
plot = sns.pointplot(x='Difference', y='Sample Size', hue='alpha', ax=ax, data=df)
ax.set_title('50% p1: Sample Size vs Proportion Difference (by Significance Level)', fontsize=18)
ax.set_xlabel('Difference (%)', fontsize=14)
ax.set_ylabel('Sample Size', fontsize=14)
plt.show()
```

![maximum sample size](https://github.com/monstott/Blogs/raw/master/Blog7/maxSize2.png)

**Observations**:
* The required sample size decreases dramatically with change in proportion difference of 1% to 2%.
* Large proportion differences require very small sample sizes.
* As the sample proportion difference increases, the difference between significance levels becomes negligible.

### Final Thoughts.

This investigation highlighted the relationship between many concepts in statistical hypothesis testing. In particular, how sample size, effect size, significance level, and power level are interlinked. The dependencies of these power analysis parameters were tested in an example involving the Student's t-Test. This process was then contrasted with the determination of required sample sizes in an example using the two proportion Z-Test. The set of related variables in this second application included sample size, significance level, effect size, and sample proportion values. Together, these examples provide a lesson on how variables can read across domains of data. The quest for statistical significance is relevant for real values, proportions, and beyond; and, it is rooted in fundamental concepts. The insights from the statistics and visualizations of this investigation will help others and I obtain statistical significant results from questions of statisitcal inference in future experiments.
