import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,  Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
import logging
import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np

class TitanicNN(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=8))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        pass

    def __data_cleaning(self, df):
        """
        does pre-processing to data before model is fit
        we drop columns that we will not use, and
        fill NaN values with the mode from whichever
        column it is in
        :param df: dataframe to be cleaned
        :return: dataframe altered as in the method description above
        """
        logger.debug("starting data cleaning")

        # split the label column off
        # y = to_categorical(df.Survived)
        y = df.Survived
        df.drop(['Survived'], inplace=True, axis=1)

        # drop columns we are not going to use, and fill NaN's with
        # most frequently occuring value for columns we retain
        for col_name in df.columns:
            if col_name not in['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Survived']:
                df.drop([col_name], inplace=True, axis=1)
            else:
                df[col_name].fillna(df[col_name].mode()[0], inplace=True)

        logger.debug("retained column statistics\n%s" % df.describe(include='all'))
        logger.debug("column datatypes:\n%s" % df.dtypes)

        # one-hot encode sex
        df = df.join(pd.get_dummies(df['Sex']))

        # drop encoded columns
        df.drop(['Sex'], axis=1, inplace=True)

        # label encode cabin
        le = LabelEncoder()
        df['Cabin'] = le.fit_transform(df['Cabin'])

        logger.debug('column names after encoding categorical values:\n%s' % df.columns)
        logger.debug('sample of transformed data:\n%s' % df.sample(n=5))

        # scale dataframe
        return (df - df.mean()) / (df.max() - df.min()), y

    def train(self, train_csv_name):
        x, y = self.__data_cleaning(pd.read_csv(train_csv_name))

        logger.debug('sample of features data frame after cleaning completed:\n%s' % x.sample(n=5))
        logger.debug('sample of labels:\n%s' % y[np.random.randint(y.shape[0], size=5)])

        logger.debug('starting model fitting')
        self.model.fit(x, y, epochs=20, batch_size=20)

    def predict(self, test_csv_name):
        pass

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
    titanicMlp = TitanicNN()
    titanicMlp.train(args.train_data)