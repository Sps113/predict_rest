import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
# import matplotlib.pyplot as plt
import werkzeug
from flask import render_template, make_response, Flask, request, url_for, redirect, flash, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.utils import secure_filename
# from image_quality_assessment.src.evaluater.predict import predict
import sys
import subprocess
from os import path
# sys.path.append(path.abspath('image_quality_assessment/src'))
# from evaluater.predict import predict
import os
import pandas as pd
import ast

from pprint import pprint
app = Flask(__name__)
api = Api(app)
uploads_dir = 'uploads'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'


class Category(Resource):
    def get(self):
        resp = make_response(render_template('map.html'), 200)
        return resp

    def post(self):

        parser = reqparse.RequestParser()  # initialize

        parser.add_argument('x_coord', required=True)  # add args
        parser.add_argument('y_coord', required=True)
        parser.add_argument('year', required=True)
        parser.add_argument('nearest1', required=True)
        parser.add_argument('nearest2', required=True)


        args = parser.parse_args()
        x_coord = float(args['x_coord'])
        x = 2537790 - 51943.4 * x_coord

        y_coord = float(args['y_coord'])
        y = 4546360 - 1.00002 * y_coord
        year = int(args['year'])
        nearest1 = int(args['nearest1'])
        nearest2 = int(args['nearest2'])

        # Load the dataset for training and testing
        data = pd.read_csv(os.getcwd() + "/data2.csv")
        # Define a field of function y = W*x + a, y = categories, x = features
        categories = data['Categories']

        # Define fields of variables

        numeric_features_list = ["x_coord", "y_coord", "Year" , "nearest1", "nearest2"
                                 #"Area (m2)", "Population", "Female", "Male", "Population Density",
                                 #"Household (person)",
                                 #"Per Capita Income", "Housing Unit Prices For Sale", "Housing Unit Prices For Rent",
                                 #"Commercial Property Unit Prices For Sale",
                                 #"Commercial Property Unit Prices For Rent"
                                 ]

        numeric_features = data[numeric_features_list]

        # print(numeric_features)

        # Normalization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(numeric_features)
        normalized_data = pd.DataFrame(scaler.transform(numeric_features), columns=numeric_features_list)

        # Split dataset into training and testing subsets
        from sklearn.model_selection import train_test_split

        # Shuffle and split the data
        X_train, X_test, y_train, y_test = train_test_split(numeric_features, categories, test_size=0.10,
                                                            random_state=30)
        # Imbalanced

        from imblearn.over_sampling import RandomOverSampler

        # remove warning
        import warnings
        warnings.filterwarnings('ignore')

        ros = RandomOverSampler(random_state=101)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        # K-neighbors classifier

        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier()

        dtc.fit(X_train, y_train)
        Y_pred = dtc.predict(X_test)
        Y_test = y_test

        # check geo
        from shapely.geometry import Point
        from shapely.geometry.polygon import Polygon

        # neighbourhood (FEVZİ ÇAKMAK, MENDERES, MİMAR SİNAN, NİNE HATUN, KAZIM KARABEKİR)
        fevzi = Polygon([(28.875267, 41.051028),
                            (28.877421, 41.047655),
                            (28.879829, 41.043863),
                            (28.8813776, 41.0408277),
                            (28.882416,	41.037665),
                            (28.886308, 41.036875),
                            (28.883373, 41.040463),
                            (28.886084, 41.041553),
                            (28.884737, 41.045517),
                            (28.881917, 41.046059),
                            (28.882140, 41.048434),
                            (28.879354, 41.049856),
                            (28.878233, 41.049436),
                            (28.877826, 41.050082)])
        menderes = Polygon([(28.871350, 41.041570),
                            (28.8716948, 41.0412201),
                            (28.8718450, 41.0408155),
                            (28.8721293, 41.0396664),
                            (28.8722420, 41.0391040),
                            (28.872570, 41.038549),
                            (28.878304, 41.039085),
                            (28.882416, 41.037665),
                            (28.879829, 41.043863),
                            (28.874927, 41.045388),
                            (28.874794, 41.043424)])
        mimar = Polygon([(28.8825524, 41.0374693),
                            (28.8840008, 41.0357132),
                            (28.885477, 41.033995),
                            (28.8895047, 41.0339814),
                            (28.893564, 41.034137),
                            (28.892879, 41.034887),
                            (28.886308, 41.036875)])
        nine = Polygon([(28.872570, 41.038549),
                            (28.876524, 41.031928),
                            (28.885477, 41.033995),
                            (28.882677, 41.037488),
                            (28.882416, 41.037665),
                            (28.878304, 41.039085)])
        kazim = Polygon([(28.877421, 41.047655),
                            (28.874646, 41.049196),
                            (28.871943, 41.046696),
                            (28.869298, 41.048241),
                            (28.868266, 41.047029),
                            (28.868491, 41.046937),
                            (28.867709, 41.045926),
                            (28.868806, 41.045017),
                            (28.868825, 41.044053),
                            (28.871350, 41.041570),
                            (28.874794, 41.043424),
                            (28.874927, 41.045388),
                            (28.879829, 41.043863)])

        if (fevzi.contains(Point(x_coord, y_coord))):
            dop = [487415.2707, 32.578, 15.983, 16.595, 0.0668, 3.89, 4.964, 3.507, 14, 4.182, 26]
            neighborhood = "FEVZİ ÇAKMAK NEIGHBORHOOD"
        elif (menderes.contains(Point(x_coord, y_coord))):
            dop = [424231.3926, 35.958, 17.732, 18.226, 0.0848, 3.84, 4.863, 3.360, 13, 3.205, 20]
            neighborhood = "MENDERES NEIGHBORHOOD"
        elif (mimar.contains(Point(x_coord, y_coord))):
            dop = [441781.8782, 32.364, 16.019, 16.345, 0.0733, 3.84, 5.001, 3.618, 14, 4.366, 24]
            neighborhood = "MIMAR SINAN NEIGHBORHOOD"
        elif (nine.contains(Point(x_coord, y_coord))):
            dop = [500835.4132, 41.827, 20.654, 21.173, 0.0835, 3.76, 5.000, 3.200, 13, 5.356, 26]
            neighborhood =  "NENE HATUN NEIGHBORHOOD"
        elif (kazim.contains(Point(x_coord, y_coord))):
            dop = [173346.8904, 13.279, 6.556, 6.723, 0.0766, 3.51, 5.193, 3.507, 14, 4.370, 25]
            neighborhood = "KAZIM KARABEKIR NEIGHBORHOOD"
        else:
            return {'error': "x:" + str(x_coord) + " y:" + str(y_coord) + ' - Place not in our Neighbourhoods'}, 200

        input_total =  [x_coord] + [y_coord] + [year] + [nearest1] + [nearest2]
        data = dtc.predict([input_total])  # convert dataframe to dictionary
        return {'category': data[0].tolist(), 'neighborhood': neighborhood}, 200  # return data and 200 OK code

api.add_resource(Category, '/geo')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
