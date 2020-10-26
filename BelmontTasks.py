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

    Time_without = [148.51, 146.65, 148.52, 150.7, 150.42, 150.88, 151.57, 147.54,
                    149.65, 148.74, 147.86, 148.75, 147.5, 148.26, 149.71, 146.56,
                    151.19, 147.88, 149.16, 148.82, 148.96, 152.02, 146.82, 149.97,
                    146.13, 148.1, 147.2, 146, 146.4, 148.2, 149.8, 147,
                    147.2, 147.8, 148.2, 149., 149.8, 148.6, 146.8, 149.6,
                    149., 148.2, 149.2, 148., 150.4, 148.8, 147.2, 148.8,
                    149.6, 148.4, 148.4, 150.2, 148.8, 149.2, 149.2, 148.4,
                    150.2, 146.6, 149.8, 149., 150.8, 148.6, 150.2, 149.,
                    148.6, 150.2, 148.2, 149.4, 150.8, 150.2, 152.2, 148.2,
                    149.2, 151., 149.6, 149.6, 149.4, 148.6, 150., 150.6,
                    149.2, 152.6, 152.8, 149.6, 151.6, 152.8, 153.2, 152.4,
                    152.2]
    type(Time_without)
    # Compute mean and standard deviation: mu, sigma
    mu = np.mean(Time_without)
    sigma = np.std(Time_without)

    print('meand, std:', mu, sigma)

    h = sorted(Time_without)  # sorted

    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed

    pl.plot(h, fit, '-o')

    pl.hist(h, density=True)  # use this to draw histogram of your data

    # Make a legend, set limits and show plot
    _ = plt.xlabel('Time without outliers')
    _ = plt.ylabel('PDF')
    plt.ylim(-0.01, 0.42)
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