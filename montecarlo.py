

# Thomas Pulliam (tjp4rb)
# STAT 3120: Extra Credit Assignment
# 10/31/2018


# "Use the Monte-Carlo method to estimate the bias and MSE for the estimator we discussed in class"
# I assumed this meant the estimator for P(X>100) = 1 - Φ[(100-µ)/(σ/√n)] from slide 21 of the chapter 7.2 slides


import math
import statistics as stats
from scipy.stats import norm


# function does Monte Carlo simulation to estimate bias and MSE of our estimator for P(X>100)
# n = size of each individual sample
# mu = population mean
# sigma = population standard deviation
def monte_carlo_estimate(n, mu, sigma):
    print("------------------------------------------------------------------------------------------------------")
    print("Size of each sample:  n =", n)
    print("Population mean:  mu =", mu)
    print("Population variance:  sigma^2 =", sigma**2)

    # number of samples generated for MC simulation
    K = 10000

    # array that will contain all of our data sets from this distribution
    samples = []
    for i in range(K):
        samples.append(norm.rvs(mu, sigma ** 2, n))

    # array that will contain our calculations of the estimator for each sample
    estimators = []

    for sample in samples:

        # sample mean and variance
        mu_hat = stats.mean(sample)
        # sigma_2_hat = stats.variance(sample, xbar=mu_hat)
        sigma_2_hat = 0
        for x in sample:
            sigma_2_hat += ((x-mu_hat)**2)/n

        # calculating our estimator for this sample
        arg = (100 - mu_hat)/(math.sqrt(sigma_2_hat/n))
        estimators.append(1 - norm.cdf(arg))

    # we will use the median of all of our estimators as our final estimate for P(X>100)
    estimate = stats.mean(estimators)
    print("Estimate for P(X>100):", estimate)

    # the actual probability is 1 - F(100; mu, sigma) where F is the cdf of a normal distribution
    true_val = 1 - norm.cdf(100, mu, sigma)
    print("Actual value for P(X>100):", true_val)

    # estimated expected value and variance for the point estimator
    expectation = estimate
    variance = stats.variance(estimators)

    # bias[estimate] = E[estimate] - E[actual value]
    bias = expectation - true_val
    print("Estimated Bias:", bias)

    # MSE[estimate] = Variance[estimate] + (Bias[estimate])^2
    mse = variance + bias ** 2
    print("Estimated MSE:", mse)
    print("------------------------------------------------------------------------------------------------------")


# does above simulation with values specified by keyboard inputs
def do_monte_carlo():
    n = int(input("Enter a sample size: "))
    mu = int(input("Enter a population mean: "))
    sigma = int(input("Enter a population standard deviation: "))
    monte_carlo_estimate(n, mu, sigma)


# example simulations with different population parameters, sample sizes, and number of trials
# monte_carlo_estimate(30, 100, 10)
# monte_carlo_estimate(80, 97, 11)

do_monte_carlo()

