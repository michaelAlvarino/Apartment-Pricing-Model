
import pandas as pd
import numpy as np
import pickle
import time
from sklearn import preprocessing, linear_model, model_selection
import matplotlib.pyplot as plt

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
    # remove outliers where rent is more than 10600/month
    max_rent = Y < 10600
    X = X[max_rent]
    Y = Y[max_rent]
    # remove people who somehow don't pay rent
    min_rent = Y > 0
    X = X[min_rent]
    Y = Y[min_rent]
    # remove unreported sqrfootage... a LOT of people don't report this
    # it might be good to find a way to still include the rest of their data
    sqrft_reported = X.loc[:, 'UNITSIZE'] != -6
    X = X[sqrft_reported]
    Y = Y[sqrft_reported]
    print("Narrowed down to " + str(X.shape[0]) + " good data points")
    return X, Y

def get_descriptions():
    desc = {
        "BEDROOMS": "Number of bedrooms 0 - 10",
        "BATHROOMS": "1 bathroom - '01', 1.5 bathroom - '02', 2 bathroom - '03' ... etc",
        "UNITSIZE": "Square footage of unit",
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
        "STORIES": "Stories in structure",
        "UTILAMT": "Amount paid towards utilities",
        "INSURAMT": "Amount paid towards renters or homeowners insurance",
        "ELECAMT": "Amount paid towards electricity",
        "GASAMT": "Amount paid towards gas",
        "WATERAMT": "Amount paid towards water",
        "HOTWATER": "Water heating fuel"
    }
    return desc

def trend(X, y):
    desc = get_descriptions()
    for colname in X:
        plt.plot(X.loc[:, colname], y, 'ro')
        plt.ylabel("Cost")
        plt.xlabel(desc[colname])
        plt.savefig("output/" + colname + ".png")
        plt.close()
    plt.hist(y, bins=40)
    plt.savefig("output/y_distribution.png")
    plt.close()
    plt.hist(X.loc[:, 'UNITSIZE'], bins=40)
    plt.savefig("output/sqrft_distribution.png")
    plt.close()

def regress(X, y):
    model = linear_model.BayesianRidge(normalize=True)
    model.fit(X, y)
    return model

def add_poly(X, deg=6):
    poly = preprocessing.PolynomialFeatures(deg)
    return poly.fit_transform(X)

def main():
    # y values cap out at 10600??
    X, y = clean('./household.csv')
    #print("Creating trends")
    #trend(X, y)
    best_score = 99999
    for degree in range(7):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
        print("Adding polynomial features of degree "+ str(degree))
        X_train = add_poly(X_train, deg=degree)
        print("Training model")
        start = time.time()
        model = regress(X_train, y_train)
        end = time.time()
        print("Total training time in seconds: ", str(end - start))
        X_test = add_poly(X_test, deg=degree)
        score = model.score(X_test, y_test)
        print("Score: " + str(score))
        print("For values: " + str(X_test[0, :]) + " the model predicts " + str(model.predict(X_test[0, :])) + " when the actual value is " + str(y_test.iloc[0]))
        if np.abs(1 - score) < best_score:
            best_score = np.abs(1 - score)
            best_model = model
            best_degree = degree
        print("====================================================")
    with open("./output/model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)
    print("Best model score: " + str(best_score))
    print("Best model degree: " + str(best_degree))



if __name__ == "__main__":
    # print all the colunms and rows when i ask for them! For debugging...
    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 999
    main()