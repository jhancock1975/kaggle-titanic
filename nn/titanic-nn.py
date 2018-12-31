import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,  Dropout
from keras.optimizers import SGD
import logging
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import constant

class TitanicNN(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=8))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
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
        if 'Survived' in df.columns:
            # y should be this if we change to categorical_crossentropy
            # y = to_categorical(df.Survived)
            y = df.Survived
            df.drop(['Survived'], inplace=True, axis=1)
        else:
            y = None

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
        """
        fits neural network on cleaned data
        :param train_csv_name: data file to train on
        :return: this method does not return a value
        """
        x, y = self.__data_cleaning(pd.read_csv(train_csv_name))
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.1, random_state=constant.RANDOM_STATE)
        logger.debug('sample of features data frame after cleaning completed:\n%s' % x.sample(n=5))
        logger.debug('sample of labels:\n%s' % y[np.random.randint(y.shape[0], size=5)])

        logger.debug('starting model fitting')
        self.model.fit(x_train, y_train, epochs=100, batch_size=50)
        y_hat = self.model.predict(x_val)
        logger.debug('prediction accuracy on validation data: %f' %
                     accuracy_score(y_val, np.rint(y_hat)))
        pass

    def generate_submission(self, test_csv_name):
        """
        generates a submission file from test file csv
        will only function properly after a model is trained
        :param test_csv_name: path to, and name of test csv data
        :return: this method does not return a value
        """
        if self.model == None:
            logger.debug('model has value None, probably not trained yet')
            raise ValueError('model has value None, probably not trained yet')
        x, _ = self.__data_cleaning(pd.read_csv(test_csv_name))
        predictions = pd.DataFrame(np.rint(self.model.predict(x)).astype(int))
        passenger_id = pd.DataFrame(pd.read_csv(test_csv_name)['PassengerId'])
        # 'PassengerId': passenger_id,
        logger.debug('passenger_id shape:')
        logger.debug(passenger_id.shape)
        logger.debug('predictions shape:')
        logger.debug(predictions.shape)

        df = pd.concat([passenger_id, predictions], axis=1 )
        df.columns=['PassengerId', 'Survived']
        df.to_csv('../data/submission.csv', index=None)
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
    titanicMlp.generate_submission(args.test_data)