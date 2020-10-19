from dfply import *
import matplotlib.pyplot as plt
from typing import Final

def normal_pdf_example_main():
    # Declares constants.
    AMOUNT_OF_SAMPLES : Final = 100000;
    AMOUNT_OF_BINS : Final = 100;

    # Generates the sample data for generating the graphs.
    # Draws 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
    samples_std1 = np.random.normal(20, 1, size=AMOUNT_OF_SAMPLES)
    samples_std3 = np.random.normal(20, 3, size=AMOUNT_OF_SAMPLES)
    samples_std10 = np.random.normal(20, 10, size=AMOUNT_OF_SAMPLES)

    # Make histograms
    _ = plt.hist(samples_std1, bins=AMOUNT_OF_BINS, density=True, histtype='step')
    _ = plt.hist(samples_std3, bins=AMOUNT_OF_BINS, density=True, histtype='step')
    _ = plt.hist(samples_std10, bins=AMOUNT_OF_BINS, density=True, histtype='step')

    # Make a legend, set limits and show plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'))

    plt.ylim(-0.01, 0.42)
    plt.show()

    # Generate CDFs
    x_std1, y_std1 = ecdf(samples_std1)
    x_std3, y_std3 = ecdf(samples_std3)
    x_std10, y_std10 = ecdf(samples_std10)

    # Plot CDFs
    _ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
    _ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
    _ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

    # Make a legend and show the plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
    plt.show()


# ECDF function
def ecdf(data):
    # number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    return x, y


if __name__ == "__main__":
    normal_pdf_example_main()