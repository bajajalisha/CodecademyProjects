#This excersize used plyplot to look at histograms for flowers in and flights to Acadia, Maine to visualize the data

# import codecademylib3
import codecademylib3
import numpy as np
from matplotlib import pyplot as plt

# load in data
in_bloom = np.loadtxt(open("in-bloom.csv"), delimiter=",")
flights = np.loadtxt(open("flights.csv"), delimiter=",")

plt.figure(1)
plt.subplot(211)

# flight histograms
plt.hist(flights, range = (0, 365), bins = 365)

plt.title("Flights to Maine")
plt.xlabel("Day of the Year")
plt.ylabel("Flight Count")

#in bloom flowers histogram
plt.subplot(212)

plt.hist(in_bloom, range = (0, 365), bins = 365)
plt.tight_layout()
plt.show()

plt.show()
