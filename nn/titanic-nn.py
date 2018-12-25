import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import logging
import argparse

class TitanicNN(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=7))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
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
        df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Survived']]
        for col_name in df.columns:
            df[col_name].fillna(df[col_name].mode()[0], inplace=True)
        logger.debug("\n%s" % df.describe(include='all'))
        return df

    def train(self, train_csv_name):
        df = self.__data_cleaning(pd.read_csv(train_csv_name))
        X = df


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