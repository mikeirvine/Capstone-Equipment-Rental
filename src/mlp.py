import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler




def load_and_standardize_data():
    ''' loads and standardizes data '''
    df_agg = pd.read_pickle('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-Equipment-Rental-Data/df_agg.pkl')
    df_agg_test = df_agg.loc[df_agg['month'] >= '2017-05-01', :]
    df_agg_train = df_agg.loc[df_agg['month'] < '2017-05-01', :]
    X_train = df_agg_train.drop(['units_rented', 'month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).values
    y_train = df_agg_train['units_rented'].values
    X_test = df_agg_test.drop(['units_rented', 'month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).values
    y_test = df_agg_test['units_rented'].values

    standardizer = StandardScaler()
    standardizer.fit(X_train, y_train)
    X_train_std = standardizer.transform(X_train)
    X_test_std = standardizer.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test

def define_nn_mlp_model(X_train):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    model = Sequential() # sequence of layers
    num_neurons_in_layer = 500 # number of neurons in a layer (is it enough?)
    num_inputs = X_train.shape[1] # number of features (784) (keep)
    model.add(Dense(units=num_neurons_in_layer,
                    input_dim=num_inputs,
                    kernel_initializer='normal',
                    activation='relu'))

    # model.add(Dense(units=250,
    #                 input_dim=num_neurons_in_layer,
    #                 kernel_initializer='normal',
    #                 activation='relu'))

    model.add(Dense(units=1,
                    input_dim=num_neurons_in_layer,
                    kernel_initializer='normal',
                    activation='relu'))

    model.compile(optimizer='adam', loss='mse') #try adam optimizer too

    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict(X_train_std, verbose=0)
    y_test_pred = model.predict(X_test_std, verbose=0)
    print('\nRandom number generator seed: {}'.format(rng_seed))
    print('\nFirst 30 labels:      {}'.format(y_train[:30]))
    print('First 30 predictions: {}'.format(y_train_pred[:30]))

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    print('MLP RMSE train results: {}'.format(train_rmse))

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    print('MLP RMSE test results: {}'.format(test_rmse))


if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    X_train_std, y_train, X_test_std, y_test = load_and_standardize_data()
    np.random.seed(rng_seed)
    model = define_nn_mlp_model(X_train_std)
    # Hmm, the fit uses 5 epochs with a batch_size of 5000.  I wonder if that's best?
    model.fit(X_train_std, y_train, epochs=200, batch_size=125, verbose=1,
              validation_split=0.1) # cross val to estimate test error, can monitor overfitting
    print_output(model, y_train, y_test, rng_seed)
