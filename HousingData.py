
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
'''
Note the following from 
https://www.census.gov/programs-surveys/ahs/data/interactive/docs/2013%20v%202015%20Metro%20Areas.pdf

New York:

2015 AHS data are not comparable to 2013 AHS data because 2015 includes Essex County, NJ,
Hunterdon County, NJ, Morris County, NJ, Somerset County, NJ, Sussex County, NJ, Union County, NJ,
Bergen County, NJ, Hudson County, NJ, Middlesex County, NJ, Monmouth County, NJ, Ocean County, NJ,
Passaic County, NJ, Dutchess County, NY, and Pike County, PA, which were not included in 2013.

So the data set that I have includes Hudson County NJ as well as other nearby NJ areas (including a PA address?) 

'''
def clean(fname):
    # read the data out of csvs
    data = pd.read_csv(fname)
    # NY OMB13CBSA = 35620
    NY = data[data['OMB13CBSA'] == "\'35620\'"]
    # this all takes one value... just remove it
    del NY['OMB13CBSA']
    print("Found " + str(NY.shape[0]) + " NY entries")
    X = NY.loc[:, ['BEDROOMS', 'DINING', 'FINROOMS', 'KITCHSINK', 'COOKTYPE', 'STORIES',
    'UTILAMT', 'ELECAMT', 'GASAMT', 'WATERAMT']]
    Y = NY.loc[:, 'RENT']
    X.loc[:, 'BATHROOMS'] = NY.loc[:, 'BATHROOMS'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'COOKTYPE'] = NY.loc[:, 'COOKTYPE'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'KITCHSINK'] = NY.loc[:, 'KITCHSINK'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'UNITSIZE'] = NY.loc[:, 'UNITSIZE'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'PORCH'] = NY.loc[:, 'PORCH'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'FIREPLACE'] = NY.loc[:, 'FIREPLACE'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'HEATTYPE'] = NY.loc[:, 'HEATTYPE'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'ACPRIMARY'] = NY.loc[:, 'ACPRIMARY'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'FRIDGE'] = NY.loc[:, 'FRIDGE'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'HOTWATER'] = NY.loc[:, 'HOTWATER'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'WASHER'] = NY.loc[:, 'WASHER'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'DRYER'] = NY.loc[:, 'DRYER'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'DISHWASH'] = NY.loc[:, 'DISHWASH'].map(lambda x: int(x[1:-1]))
    X.loc[:, 'BLD'] = NY.loc[:, 'BLD'].map(lambda x: int(x[1:-1]))
    return X, Y

def get_descriptions():
    desc = {
        "BEDROOMS": "Number of bedrooms 0 - 10",
        "BATHROOMS": "1 bathroom - '01', 1.5 bathroom - '02', 2 bathroom - '03' ... etc",
        "UNITSIZE_IUF": "Square footage of unit",
        "PORCH": "Porch, deck, balcony or patio",
        "FIREPLACE": "Usable fireplace",
        "DINING": "Separate dining room",
        "FINROOMS": "With 2 or more living rooms or recreation rooms, etc",
        "HEATTYPE": "Main Heating Equipment",
        "ACPRIMARY": "Primary Air Conditioning",
        "FRIDGE": "Refrigerator",
        "KITCHSINK": "Kitchen sink",
        "COOKTYPE": "Cooking stove or range",
        "DISHWASH": "Dishwasher",
        "WASHER": "Clothes washing machine",
        "DRYER": "Clothes dryer",
        "BLD": "Number of units in structure",
        "STORIES_IUF": "Stories in structure",
        "UTILAMT": "Amount paid towards utilities",
        "INSURAMT": "Amount paid towards renters or homeowners insurance",
        "ELECAMT": "Amount paid towards electricity",
        "GASAMT": "Amount paid towards gas",
        "WATERAMT": "Amount paid towards water"
    }
    return desc

def main():
    Lambda = 1
    X, y = clean('./household.csv')


if __name__ == "__main__":
    # print all the colunms when i ask for them! For debugging...
    pd.options.display.max_columns = 999
    main()