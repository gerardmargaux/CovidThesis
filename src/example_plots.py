# generates the examples plots written in the latex file
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import tensorflow as tf

import random
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

plot_example_dir = '../plot/examples'
n_samples = 20
n_forecast = 10
sample_test = 30  # number of test sample predicted on for the prediction on a horizon
sample_train = 150  # number of training sample points used by a trainable model

color_train = '#1f77b4'
color_prediction = '#ff7f0e'
color_actual = '#2ca02c'
steps = np.pi / 30  # steps used between 2 points


def target_function(x):  # target used for the models
    return (np.cos(x) + 1) / 2


def plot_sample_prediction(y_train, y_predicted, y_actual):
    """
    function called by all plotting function for a single window. Plot the target used in the training part of the
    window (y_train), the target predicted (y_predicted) and the real target (y_actual)
    """
    plt.figure(figsize=(5, 4))
    x_train = range(len(y_train))
    x_predicted = range(len(y_train), len(y_train) + len(y_predicted))
    plt.plot(x_train, y_train, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(x_predicted, y_actual, 'o', color=color_actual, label='True value')
    plt.plot(x_predicted, y_predicted, 'X', color=color_prediction, label='Prediction')
    plt.grid()
    plt.legend()
    plt.tight_layout()


def plot_prediction_single_horizon(y_predicted, y_actual, y_train=None, horizon=1):
    """
    plot the prediction on the given horizon
    :param y_predicted: np.array of predicted values (shape=(sample, n_forecast))
    :param y_actual: np.array of actual values (shape=(sample, n_forecast))
    :param horizon: horizon that should be plotted
    :param y_train: np.array of training values that should be shown (None= no training values shown)
    """
    plt.figure(figsize=(5, 4))
    if y_train is None:
        y_train = []
    # plot the last y_actual points as the beginning of the curve
    y_predicted = y_predicted[:, horizon-1].flatten()
    y_actual = y_actual[:, horizon-1].flatten()
    x_train = range(len(y_train))
    x_predicted = range(len(y_train), len(y_train) + len(y_predicted))
    if len(y_train) > 0:
        plt.plot(x_train, y_train, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(x_predicted, y_actual, linestyle='-', marker='o', color=color_actual, label=f'True value t+{horizon}')
    plt.plot(x_predicted, y_predicted, linestyle='-', marker='X', color=color_prediction, label=f'Prediction t+{horizon}')
    plt.grid()
    plt.legend()
    plt.tight_layout()


def plot_prediction_reference_models():  # plot the predictions on a single window
    X_values = np.arange(n_samples + n_forecast) * steps
    values = target_function(X_values)
    y_train = values[:n_samples]
    y_test = values[n_samples:]
    models = [prediction_linear_regression, prediction_baseline]
    for model in models:
        prediction = model(y_train, n_forecast)
        plot_sample_prediction(y_train, prediction, y_test)
        plt.savefig(f'{plot_example_dir}/{model.__name__}', dpi=200)


def plot_prediction_dense_model():
    # test set
    X_values = np.arange(n_samples + n_forecast) * steps
    values = target_function(X_values)
    X_test = values[:n_samples].reshape((1, n_samples))
    Y_test = values[n_samples:].reshape((1, n_forecast))
    X_train = np.zeros((sample_train, n_samples))
    Y_train = np.zeros((sample_train, n_forecast))
    # training set
    delta = 0.01  # little variation from the test set
    for j, i in enumerate(range(n_samples + n_forecast, n_samples + n_forecast + sample_train)):
        X_values = np.arange(i, i + n_samples + n_forecast) * steps + delta
        values = target_function(X_values)
        X_train[j, :] = values[:n_samples].reshape((1, n_samples))
        Y_train[j, :] = values[n_samples:].reshape((1, n_forecast))
    prediction = prediction_dense_model(X_train, Y_train, X_test)
    plot_sample_prediction(X_test.flatten(), prediction.flatten(), Y_test.flatten())
    plt.savefig(f'{plot_example_dir}/prediction_dense_model', dpi=200)


def plot_prediction_reference_models_t_1():  # plot the predictions on a single horizon
    X_values = np.zeros((sample_test, n_samples))
    y_test = np.zeros((sample_test, n_forecast))
    for i in range(sample_test):
        x_range = np.arange(i, i + n_samples) * steps
        y_range = np.arange(i + n_samples, i + n_samples + n_forecast) * steps
        X_values[i, :] = target_function(x_range)
        y_test[i, :] = target_function(y_range)
    models = [prediction_linear_regression, prediction_baseline]
    for model in models:
        prediction = np.zeros((sample_test, n_forecast))
        for i in range(sample_test):
            prediction[i, :] = model(X_values[i, :], n_forecast)
        plot_prediction_single_horizon(prediction, y_test, y_train=None, horizon=1)
        plt.savefig(f'{plot_example_dir}/{model.__name__}_t_1', dpi=200)


def plot_prediction_dense_model_t_1():
    # test set
    X_values = np.arange(n_samples + n_forecast) * steps
    values = target_function(X_values)
    # X_test = values[:n_samples].reshape((1, n_samples))
    # Y_test = values[n_samples:].reshape((1, n_forecast))
    X_train = np.zeros((sample_train, n_samples))
    Y_train = np.zeros((sample_train, n_forecast))
    X_test = np.zeros((sample_test, n_samples))
    Y_test = np.zeros((sample_test, n_forecast))
    # training set
    delta = 0.01  # little variation from the test set
    for i in range(sample_test):
        x_range = np.arange(i, i + n_samples) * steps
        y_range = np.arange(i + n_samples, i + n_samples + n_forecast) * steps
        X_test[i, :] = target_function(x_range)
        Y_test[i, :] = target_function(y_range)
    for j, i in enumerate(range(sample_test + n_forecast + n_samples, sample_test + n_forecast + n_samples + sample_train)):
        X_values = np.arange(i, i + n_samples + n_forecast) * steps + delta
        values = target_function(X_values)
        X_train[j, :] = values[:n_samples].reshape((1, n_samples))
        Y_train[j, :] = values[n_samples:].reshape((1, n_forecast))
    prediction = prediction_dense_model(X_train, Y_train, X_test)
    plot_prediction_single_horizon(prediction, Y_test, y_train=None, horizon=1)
    plt.savefig(f'{plot_example_dir}/prediction_dense_model_t_1', dpi=200)


def prediction_linear_regression(x_train, nb_test):
    axis = np.arange(len(x_train)).reshape(-1, 1)
    regr = LinearRegression().fit(axis, x_train)
    return regr.predict(np.arange(len(x_train), len(x_train) + nb_test).reshape(-1, 1))


def prediction_baseline(x_train, nb_test):
    return np.full(nb_test, x_train[-1])


def get_dense_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(n_forecast)])
    model.compile(loss=tf.losses.MeanSquaredError())
    return model


def prediction_dense_model(X_train, Y_train, X_test, epochs=150):
    # create and train the model
    model = get_dense_model()
    model.fit(X_train, Y_train, verbose=0, epochs=epochs)
    prediction = model.predict(X_test)
    return prediction


def plot_prediction():
    plot_prediction_reference_models()
    plot_prediction_dense_model()


def plot_prediction_t_1():
    plot_prediction_reference_models_t_1()
    plot_prediction_dense_model_t_1()


def random_query():
    geo = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
    'GB': 'Great Britain',
    'GR': 'Greece',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IS': 'Iceland',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
}
    geo_list = list(geo.keys())  # random.choice does not work on dic
    topics = {
        'Fièvre': '/m/0cjf0',
        'Mal de gorge': '/m/0b76bty',
        # 'Dyspnée': '/m/01cdt5',
        # 'Agueusie': '/m/05sfr2',
        # 'Anosmie': '/m/0m7pl',
        # 'Virus': '/m/0g9pc',
        # 'Épidémie': '/m/0hn9s',
        'Symptôme': '/m/01b_06',
        # 'Thermomètre': '/m/07mf1',
        # 'Grippe espagnole': '/m/01c751',
        # 'Paracétamol': '/m/0lbt3',
        # 'Respiration': '/m/02gy9_',
        # 'Toux': '/m/01b_21',
        # 'Coronavirus': '/m/01cpyy'
    }
    kw = [[code] for code in topics.values()]

    def random_timeframe():
        end_date = datetime.date(year=random.randint(2006, 2020), month=random.randint(1, 12), day=random.randint(1, 28))
        delta = datetime.timedelta(days=random.randint(8, 270))
        beign_date = end_date - delta
        timeframe = f"{beign_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        return timeframe

    for loc in random.choice(geo_list):
        for search in random.choice(kw):
            pytrends = TrendReq()
            pytrends.build_payload(search, cat=0, timeframe=random_timeframe(), geo=loc)
            df = pytrends.interest_over_time()




if __name__ == '__main__':
    plot_prediction()
    plot_prediction_t_1()
