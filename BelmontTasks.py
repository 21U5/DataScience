import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab as pl

#ECDF function
def ecdf(data):
    # number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x=np.sort(data)
    # y-data for the ECDF: y
    y=np.arange(1, n+1)/n
    return x,y

def belmont_tasks_main():
    # Task: create a function to convert time to milliseconds (Done: Used custom function convert_time_value_to_seconds to convert the string values to seconds.)
    belmont = pd.read_csv("belmont.csv", sep = ';', decimal= ",") # use delimeters, try to check if all data are available

    time_column = belmont["Time"]

    for i in range(len(time_column)):
        time_column[i] = convert_time_value_to_seconds(time_column[i])

    print(time_column)

    mu_mean = np.mean(time_column)
    sigma_std = np.std(time_column)
    print('mean', mu_mean)
    print('standard deviation', sigma_std)

    # calculate Z-scores to find outliers
    # Task: calc outliers using another methods see reccomandations (Done: Using boxplotting.)

    plt.boxplot(time_column)

    plt.show()

    # 2 Outliers discovered: the min and the max value.

    Outliers_min = time_column.min()
    Outliers_max = time_column.max()

    # Displays the outliers.

    print('Outliers_min', Outliers_min)
    print('Outliers_max', Outliers_max)

    h_1 = sorted(time_column)  # sorted

    fit = stats.norm.pdf(h_1, np.mean(h_1), np.std(h_1))  # this is a fitting indeed

    pl.plot(h_1, fit, '-o')

    pl.hist(h_1, density=True)  # use this to draw histogram of your data

    # Make a legend, set limits and show plot
    _ = plt.xlabel('Z Scores')
    _ = plt.ylabel('PDF')
    plt.ylim(-0.01, 0.42)
    plt.show()

    # Task 1: convert to array and print the whole array, combine two columns Time and Z-scores and print as a table (Done: See below.)
    Z_scores = (time_column - mu_mean) / sigma_std
    for i in range(len(time_column)):
        print(time_column[i].__str__() + " , " + Z_scores[i].__str__())

    # Task 2: delete outliers from the Time and print

    # Converts the time_column array into a list.
    time_column_as_list = time_column.tolist()

    # Removes the min and max value (the outliers).
    time_column_as_list.remove(Outliers_min)
    time_column_as_list.remove(Outliers_max)

    time_column = np.array(time_column_as_list)

    #  a normal distribution with outliers
    h = sorted(time_column)  # sorted

    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed

    pl.plot(h, fit, '-o')

    pl.hist(h, density=True)  # use this to draw histogram of your data

    # Make a legend, set limits and show plot
    _ = plt.xlabel('Time')
    _ = plt.ylabel('PDF')
    plt.ylim(-0.01, 0.42)
    plt.show()

    # a normal distribution without outliers

    Time_without = [ 875.5,  648.2,  788.1,  940.3,  491.1,  743.5,  730.1,  686.5,
        878.8,  865.6,  654.9,  831.5,  798.1,  681.8,  743.8,  689.1,
        752.1,  837.2,  710.6,  749.2,  967.1,  701.2,  619. ,  747.6,
        803.4,  645.6,  804.1,  787.4,  646.8,  997.1,  774. ,  734.5,
        835. ,  840.7,  659.6,  828.3,  909.7,  856.9,  578.3,  904.2,
        883.9,  740.1,  773.9,  741.4,  866.8,  871.1,  712.5,  919.2,
        927.9,  809.4,  633.8,  626.8,  871.3,  774.3,  898.8,  789.6,
        936.3,  765.4,  882.1,  681.1,  661.3,  847.9,  683.9,  985.7,
        771.1,  736.6,  713.2,  774.5,  937.7,  694.5,  598.2,  983.8,
        700.2,  901.3,  733.5,  964.4,  609.3, 1035.2,  718. ,  688.6,
        736.8,  643.3, 1038.5,  969. ,  802.7,  876.6,  944.7,  786.6,
        770.4,  808.6,  761.3,  774.2,  559.3,  674.2,  883.6,  823.9,
        960.4,  877.8,  940.6,  831.8,  906.2,  866.5,  674.1,  998.1,
        789.3,  915. ,  737.1,  763. ,  666.7,  824.5,  913.8,  905.1,
        667.8,  747.4,  784.7,  925.4,  880.2, 1086.9,  764.4, 1050.1,
        595.2,  855.2,  726.9,  785.2,  948.8,  970.6,  896. ,  618.4,
        572.4, 1146.4,  728.2,  864.2,  793. ]
    type(Time_without)
    ecdf(Time_without)
    # Compute mean and standard deviation: mu, sigma
    mu = np.mean(Time_without)
    sigma = np.std(Time_without)

    print('meand, std:', mu, sigma)

    h = sorted(Time_without)  # sorted

    fit = stats.norm.cdf(h, np.mean(h), np.std(h))  # this is a fitting indeed

    pl.plot(h, fit, '-o')

    pl.hist(h, density=True)  # use this to draw histogram of your data

    # Make a legend, set limits and show plot
    _ = plt.xlabel('Time without outliers')
    _ = plt.ylabel('PDF')
    plt.ylim(-0.01, 1)
    plt.show()

    # Sample out of a normal distribution with this mu and sigma: samples
    samples = np.random.normal(mu, sigma, size=10000)

    # Get the CDF of the samples and of the data
    x_theor, y_theor = ecdf(samples)
    x, y = ecdf(Time_without)

    # Plot the CDFs and show the plot
    _ = plt.plot(x_theor, y_theor)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('Belmont winning time (sec.)')
    _ = plt.ylabel('CDF')
    plt.show()

    # to find out if it is possible to hit Secretariats record
    # Take a million samples out of the Normal distribution: samples
    samples = np.random.normal(mu, sigma, size=1000000)

    # Compute the fraction that are faster than 144 seconds: prob
    prob = np.sum(samples <= 144) / len(samples)

    # Print the result
    print('Probability of besting Secretariat:', prob)


def convert_time_value_to_seconds(time_string : str) -> float:
    minutes = int(time_string.rsplit(':', 1)[0])
    seconds = float(time_string.rsplit(':', 1)[1])
    return minutes * 60 + seconds


if __name__ == "__main__":
    belmont_tasks_main()