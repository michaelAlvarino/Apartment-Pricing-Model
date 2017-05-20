import pickle
from sklearn import preprocessing
import numpy as np

def predict(MODEL=None, BEDROOMS=1, BATHROOMS=1, UNITSIZE=3, PORCH=1,
            FIREPLACE=1, DINING=0, FINROOMS=1, HEATTYPE=1,
            ACPRIMARY=1, FRIDGE=1, KITCHSINK=1, COOKTYPE=1,
            DISHWASH=0, WASHER=0, DRYER=0, BLD=8, STORIES=4,
            UTILAMT=64, ELECAMT=60, GASAMT=4, WATERAMT=4,
            HOTWATER=1):
    model = MODEL
    if model == None:
        with open('./output/model.pkl', 'rb') as f:
            model = pickle.load(f)

    poly = preprocessing.PolynomialFeatures(model["degree"])
    X = poly.fit_transform(np.array([BEDROOMS, DINING, FINROOMS, KITCHSINK,
                                    STORIES, UTILAMT, ELECAMT, GASAMT,
                                    WATERAMT, BATHROOMS, COOKTYPE, UNITSIZE,
                                    PORCH, FIREPLACE, HEATTYPE, ACPRIMARY,
                                    FRIDGE, HOTWATER, WASHER, DRYER,
                                    DISHWASH, BLD]))
    return model["model"].predict(X)

