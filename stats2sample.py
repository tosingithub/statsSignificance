import pandas as pd
import numpy as np
import os
#import seaborn as sns
import scipy
#import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def cohen_d(group1, group2):
    """Calculate and Define Cohen's d"""
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    
    return d

def calc_variance(sample):
    '''Computes the variance a list of values'''
    sample_mean = np.mean(sample)
    return sum([(i - sample_mean)**2 for i in sample])


def sample_variance(sample1, sample2):
    '''Computes the pooled variance 2 lists of values, using the calc_variance function'''
    n_1, n_2 = len(sample1), len(sample2)
    var1, var2 = calc_variance(sample1), calc_variance(sample2)
    return (var1 + var2) / ((n_1 + n_2) - 2)


def twosample_tstatistic(expr, ctrl):
    '''Computes the 2-sample T-stat of 2 lists of values, using the calc_sample_variance function'''
    expr_mean, ctrl_mean = np.mean(expr), np.mean(ctrl)
    n_e, n_c = len(expr), len(ctrl)
    samp_var = sample_variance(expr,ctrl)
    t = (expr_mean - ctrl_mean) / np.sqrt(samp_var * ((1/n_e)+(1/n_c)))
    return t

def read_results(filename):
    result = []
    dir_list = os.listdir("./")
    if filename in dir_list:
        doc = open(filename, "r")
        for row in doc:
            cvet = [float(i) for i in row.split(" ")]
            result.append(cvet)
    return result

ALPHA = 0.05
POWER = 1

if __name__=="__main__":
    results = read_results("results.txt")
    sample1_np = np.array(results[0])
    sample2_np = np.array(results[1])

    # Cohen's d
    effect = abs(cohen_d(sample1_np, sample2_np))
    print("Effect size: ", effect)
    ratio = len(sample1_np) / len(sample2_np)
    # Perform power analysis
    analysis = TTestIndPower()
    result = int(analysis.solve_power(effect, power=POWER, nobs1=None,ratio=ratio, alpha=ALPHA))
    print(f"The minimum sample size: {result}")

    # bootstrapping: 10,000 samples of 50 (result) items each
    sample1_np_b = []
    sample2_np_b = []
    for _ in range(10000):
        sample_mean = np.random.choice(sample1_np, size=result).mean()
        sample1_np_b.append(sample_mean)
    for _ in range(10000):
        sample_mean2 = np.random.choice(sample2_np, size=result).mean()
        sample2_np_b.append(sample_mean2)

    # t statistic
    t_stat = twosample_tstatistic(sample1_np_b, sample2_np_b)
    print("T statistic: {}".format(t_stat))

    # Sanity check to confirm our t_stat
    # If our p-value below is less than or equal to our anticipated error (alpha) then we would reject our Null hypothesis.
    # The p-value is the likelihood of us getting a RANDOM result outside of our 95% confidence interval.
    sci_t = stats.ttest_ind(sample1_np_b, sample2_np_b)
    print(sci_t)
    #print(stats.ttest_ind(sample1_np_b, sample2_np_b))

    with open('sign_results.txt', "a+") as f:
        s = f.write('Effect size {}, \n The minimum sample size: {}, T statistic:  {}, \n Scipy T test: {} '.format(effect, result, t_stat, str(sci_t) + "\n\n"))
