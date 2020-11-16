import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab as pl

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

def ecdf(data):
    # number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x=np.sort(data)
    # y-data for the ECDF: y
    y=np.arange(1, n+1)/n
    return x,y


def message_main():
    for _ in range(50):
        # Generate permutation samples
        perm_sample_1, perm_sample_2 = permutation_sample(
            rain_june, rain_november)

        # Compute ECDFs
        x_1, y_1 = ecdf(perm_sample_1)
        x_2, y_2 = ecdf(perm_sample2)

        # Plot ECDFs of permutation sample
        _ = plt.plot(x_1, y1, marker='.', linestyle='none',
                   color='red', alpha=0.02)
        _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                   color='blue', alpha=0.02)


    # Create and plot ECDFs from original data
    x_1, y_1 = ecdf(rain_june)
    x_2, y_2 = ecdf(rainnovember)
    _ = plt.plot(x_1, y1, marker='.', linestyle='none', color='red')
    _ = plt.plot(x_2, y2, marker='.', linestyle='none', color='blue')

    # Label axes, set margin, and show plot
    plt.margins(0.02)
    _ = plt.xlabel('monthly rainfall (mm)')
    _ = plt.ylabel('ECDF')
    plt.show()

if __name__ == "__main__":
    message_main()