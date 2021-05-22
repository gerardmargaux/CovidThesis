# generates the examples plots written in the latex file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import util
import trends_query
import time
import random
import pandas as pd
from bisect import bisect, bisect_left

plot_example_dir = '../plot/examples'
dir_tor_experiments = '../data/trends/tor_experiment'
n_samples = 30
n_forecast = 20
sample_test = 30  # number of test sample predicted on for the prediction on a horizon
sample_train = 220  # number of training sample points used by a trainable model

color_train = '#1f77b4'
color_prediction = '#ff7f0e'
color_actual = '#2ca02c'
steps = np.pi / 30  # steps used between 2 points

# ---------------- always done at the beginning and end of each figure

def plt_prepare():
    plt.figure(figsize=(5, 4))


def plt_finish():
    plt.grid()
    plt.legend()
    plt.tight_layout()


# ---------------- plot examples of predictions

def target_function(x):  # target used for the models
    return (np.cos(x) + 1) / 2


def plot_sample_prediction(y_train, y_predicted, y_actual):
    """
    function called by all plotting function for a single window. Plot the target used in the training part of the
    window (y_train), the target predicted (y_predicted) and the real target (y_actual)
    """
    plt_prepare()
    x_train = range(len(y_train))
    x_predicted = range(len(y_train), len(y_train) + len(y_predicted))
    plt.plot(x_train, y_train, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(x_predicted, y_actual, 'o', color=color_actual, label='True value')
    plt.plot(x_predicted, y_predicted, 'X', color=color_prediction, label='Prediction')
    plt_finish()


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


# ---------------- predictions for the reference models

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


# ---------------- generate the predictions of some models

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


def plot_prediction():  # plot the predictions on a window
    plot_prediction_reference_models()
    plot_prediction_dense_model()


def plot_prediction_t_1():  # plot the predictions for t+1
    plot_prediction_reference_models_t_1()
    plot_prediction_dense_model_t_1()


# ---------------- plot for the real predictions
def real_predictions(file_pred):
    file_true = "../res/True_y_values_last_walk_BE.csv"
    prediction_df = pd.read_csv(file_pred)
    true_df = pd.read_csv(file_true)

    prediction_last_walk = prediction_df[prediction_df["Walk"] == "walk 6"]
    prediction_last_walk = prediction_last_walk[["DATE", "NEW_HOSP(t+1)"]].set_index("DATE")
    samples = true_df[:-len(prediction_last_walk)]
    samples = samples[["DATE", "NEW_HOSP(t+1)"]].set_index("DATE")
    print(samples)
    true_forecast = true_df[-len(prediction_last_walk):]
    true_forecast = true_forecast[["DATE", "NEW_HOSP(t+1)"]].set_index("DATE")

    # Plot
    fig = plt.figure(figsize=(5, 4))
    plt.plot(samples, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(true_forecast, linestyle='', marker='o', color=color_actual, label=f'True value t+1')
    plt.plot(prediction_last_walk, linestyle='', marker='X', color=color_prediction, label=f'Prediction t+1')
    ax = fig.axes[0]
    # set locator
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    # set formatter
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt_finish()
    plt.savefig(f'../plot/predictions/prediction_dense', dpi=200)


# ---------------- plot for tor


def tor_vs_local():  # comparison between tor queries and local queries
    def random_timeframe():
        end_date = date(year=random.randint(2006, 2020), month=random.randint(1, 12), day=random.randint(1, 28))
        delta = timedelta(days=random.randint(8, 260))
        beign_date = end_date - delta
        timeframe = f"{beign_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        return timeframe

    max_runtime = 3 * 3600  # runtime in seconds
    topics = util.list_topics
    geo = util.european_geocodes
    geo_list = list(geo.keys())  # random.choice does not work on dic
    sleep_intermediate = lambda: time.sleep(np.random.random())
    random.seed(int(time.time()))
    sleep_error = lambda: time.sleep(60 + np.random.randint(30, 90))
    kw = [[code] for code in topics.values()]
    pytrends_list = [trends_query.TorTrendsRequest, trends_query.LocalTrendsRequest]
    for i, pytrends_class in enumerate(pytrends_list):
        pytrends = pytrends_class(max_errors=0)
        init = time.perf_counter()
        elapsed = 0
        elapsed_nb_requests = []
        nb_requests = []
        elapsed_nb_errors = []
        nb_errors = []
        while elapsed < max_runtime:
            loc = random.choice(geo_list)
            search = random.choice(kw)
            sleep_intermediate()
            errors = pytrends.nb_exception
            try:
                pytrends.build_payload(search, cat=0, timeframe=random_timeframe(), geo=loc)
                _ = pytrends.interest_over_time()
            except Exception as err:
                print(f'caught exception ({type(err)})')
                sleep_error()
            current = time.perf_counter()
            elapsed = current - init
            elapsed_nb_requests.append(elapsed)
            nb_requests.append(pytrends.request_done)
            if pytrends.nb_exception != errors:
                elapsed_nb_errors.append(elapsed)
                nb_errors.append(pytrends.nb_exception)
            print(f'{pytrends.request_done} requests done. Elapsed time: {elapsed:.2f} [s] '
                  f'(remaining: {max_runtime-elapsed:.2f} [s]). '
                  f'({pytrends.request_done / elapsed:.3f} [req/s]). {pytrends.nb_exception} errors happened. '
                  f'Using {pytrends.__class__.__name__}')
        df_errors = pd.DataFrame(data={'elapsed': elapsed_nb_errors, 'errors': nb_errors})
        df_nb_requests = pd.DataFrame(data={'elapsed': elapsed_nb_requests, 'nb_requests': nb_requests})
        df_errors.to_csv(f'{dir_tor_experiments}/{pytrends.__class__.__name__}_errors_5.csv', index=False)
        df_nb_requests.to_csv(f'{dir_tor_experiments}/{pytrends.__class__.__name__}_nb_requests_5.csv', index=False)
        time.sleep(300)


def plot_tor_vs_local():  # plot the comparison between tor queries and local queries, using stored data
    df_errors_tor = pd.read_csv(f'{dir_tor_experiments}/TorTrendsRequest_errors_4.csv')
    df_errors_local = pd.read_csv(f'{dir_tor_experiments}/LocalTrendsRequest_errors_4.csv')
    df_nb_requests_tor = pd.read_csv(f'{dir_tor_experiments}/TorTrendsRequest_nb_requests_4.csv')
    df_nb_requests_local = pd.read_csv(f'{dir_tor_experiments}/LocalTrendsRequest_nb_requests_4.csv')
    color_tor = color_prediction
    color_local = color_train
    if not df_errors_tor.empty:
        tor_error_idx = [bisect_left(df_nb_requests_tor['elapsed'], row['elapsed'])
                       for _, row in df_errors_tor.iterrows()]
        tor_error_value = df_nb_requests_tor.iloc[tor_error_idx]['nb_requests']
        tor_error_axis = df_nb_requests_tor.iloc[tor_error_idx]['elapsed']
    if not df_errors_local.empty:
        local_error_idx = [bisect_left(df_nb_requests_local['elapsed'], row['elapsed'])
                       for _, row in df_errors_local.iterrows()]
        local_error_value = df_nb_requests_local.iloc[local_error_idx]['nb_requests']
        local_error_axis = df_nb_requests_local.iloc[local_error_idx]['elapsed']

    plt_prepare()
    # plot the last y_actual points as the beginning of the curve
    plt.plot(df_nb_requests_local['elapsed'], df_nb_requests_local['nb_requests'], color=color_local, label='local')
    if not df_errors_local.empty:
        plt.plot(local_error_axis, local_error_value, linestyle='',
                 color=color_local, marker='.', label='local error (sleep)')
    plt.plot(df_nb_requests_tor['elapsed'], df_nb_requests_tor['nb_requests'], color=color_tor, label='tor')
    if not df_errors_tor.empty:
        plt.plot(tor_error_axis, tor_error_value, linestyle='',
                 color=color_tor, marker='.', label='tor error (IP reset)')
    plt.xlabel('Time elapsed [s]')
    plt.ylabel('Number of requests [/]')
    plt_finish()
    plt.savefig(f'{plot_example_dir}/tor_vs_local_queries_4', dpi=200)
    plt_prepare()
    plt.plot(df_nb_requests_local['elapsed'], df_nb_requests_local['nb_requests'] / df_nb_requests_local['elapsed'],
             color=color_local, label='local')
    if not df_errors_local.empty:
        plt.plot(local_error_axis, local_error_value / local_error_axis, linestyle='',
                 color=color_local, marker='.', label='local error (sleep)')
    plt.plot(df_nb_requests_tor['elapsed'], df_nb_requests_tor['nb_requests'] / df_nb_requests_tor['elapsed'],
             color=color_tor, label='tor')
    if not df_errors_tor.empty:
        plt.plot(tor_error_axis, tor_error_value / tor_error_axis, linestyle='',
                 color=color_tor, marker='.', label='tor error (IP reset)')
    plt.xlabel('Time elapsed [s]')
    plt.ylabel('Rate [requests/s]')
    plt_finish()
    plt.savefig(f'{plot_example_dir}/tor_vs_local_rate_4', dpi=200)


# ---------------- plot for trends data

def plot_trends(df_plot, topic_code, show=True):
    fig = plt.figure()
    if isinstance(df_plot, list):
        list_df = df_plot
    else:
        list_df = [df_plot]
    for df in list_df:
        df_plot = 100 * df[[topic_code]] / df[[topic_code]].max()
        plt.plot(df_plot, label="hourly data")
    ax = fig.axes[0]
    # set monthly locator
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    # tor_vs_local()
    #plot_tor_vs_local()
    #df = pd.read_csv('../data/trends/model/FR-B-Fièvre.csv', parse_dates=['date']).set_index('date')
    #plot_trends(df, df.columns[0])
    file_pred = "../res/2021-05-21-16:25_get_dense_model_NEW_HOSP_prediction_BE.csv"
    real_predictions(file_pred)
    #print(df)
