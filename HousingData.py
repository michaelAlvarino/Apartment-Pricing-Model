
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt

def clean(fname):
    # read the data out of csvs
    data = pd.read_csv(fname)
    print(data.columns.values)

def main():
    Lambda = 1
    clean('./ahs2015m.csv')

if __name__ == "__main__":
    main()