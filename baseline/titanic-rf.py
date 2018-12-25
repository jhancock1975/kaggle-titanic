"""
@author jhancock1975
We wrote this code for the
Kaggle titanic contest, for beginners
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import argparse
from sklearn.metrics import accuracy_score
import constant
class Classifier(object):
    def __init__(self, train_csv_name, test_csv_name):
        """
        constructor
        :param train_csv_name: path to, and name of training data
        :param test_csv_name: path to, and name of test data
        """
        logger.debug('created %s classifier object' % self)
        self.train_csv_name = train_csv_name
        self.test_csv_name = test_csv_name
        logger.debug('training csv file name: %s' % train_csv_name)
        logger.debug('validation data file name: %s' % test_csv_name)
        self.trained_model=None

class GenderClassifier(Classifier):

    def train_and_eval(self):
        """
        uses gender as predictor for survival
        :return:
        """
        df = pd.read_csv(self.train_csv_name)
        logger.debug("gender classifier accuracy: %s" % accuracy_score(df.Survived, df.Sex=='female'))

class TitanicRf(Classifier):

    def clean_data(self, df):
        """
        clean data before training,
        for this simple case we drop any non-numeric columns, except for
        the target value, and we drop any NaN values so we can train with
        random forest
        :param df:
        :return: dataframe with cleaned data
        """
        for column_name in df.columns:
            if not np.issubdtype(df[column_name], np.number) and (column_name != 'Survived'):
                df = df.drop([column_name], axis=1)
        df.dropna(inplace=True)
        return df

    def train_and_eval(self):
        """
        trains and tests a random forest classifier, for use as the
        baseline classifier for this project
        :return:
        """
        logger.debug('starting model fitting')
        train_csv = pd.read_csv(self.train_csv_name)
        train_csv = self.clean_data(train_csv)
        X = train_csv.drop(['Survived'], axis=1)

        # the ...values.ravel() is to suppress warning
        # titanic-rf.py:67: DataConversionWarning: A column-vector y was passed when a 1d array
        # was expected. Please change the shape of y to (n_samples,), for example using ravel().
        # solution from https://stackoverflow.com/posts/36120015/revisions
        # StackOverflow.com user: Linda MacPhee-Cobb,
        # edited by StackOverflow user: Tshilidzi Mudau
        # accessed December 24th, 2018
        y = train_csv[['Survived']].values.ravel()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,
                                                          random_state=constant.RANDOM_STATE)
        logger.debug("X_train dimensions: %s", X_train.shape)
        logger.debug("X_val dimensions: %s", X_val.shape)
        logger.debug("y_train length: %s", len(y_train))
        logger.debug("y_val length: %s", len(y_val))

        rf = RandomForestClassifier(random_state=constant.RANDOM_STATE, n_estimators=10)
        logger.debug('fitting classifier')
        rf.fit(X_train, y_train)
        self.trained_model=rf

        logger.debug('starting predictions')
        predictions = rf.predict(X_val)
        logger.debug("random forest accuracy: %s" % accuracy_score(y_val, predictions))

    def test(self):
        """
        evaluates accuracy of trained model on test data
        """
        logger.debug('starting test predictions')
        test_csv = pd.read_csv(self.test_csv_name)
        test_csv = self.clean_data(test_csv)
        #test_csv = test_csv.concat(self.trained_model.predict,axis=1)
        logger.debug(test_csv.head())


# parse command line arguments
# this program expects the user
# to supply the file path, and name of
# test and training data


parser = argparse.ArgumentParser()
parser.add_argument("train_data", metavar="train-data",
                    help="path and name of csv file containing training data")
parser.add_argument("test_data", metavar="test-data",
                    help="path and name of csv file containing test data")
args = parser.parse_args()

# logging setup
logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":


    # show all the columns when printing
    pd.set_option('display.max_columns', None)

    logger.debug('starting up')
    titanicRf = TitanicRf(args.train_data, args.test_data)
    genderClassifier = GenderClassifier(args.train_data, args.test_data)
    for clf in [titanicRf, genderClassifier]:
        clf.train_and_eval()
    titanicRf.test()
