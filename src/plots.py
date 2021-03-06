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
dir_results = '../res'
n_samples = 20
n_forecast = 10
sample_test = 30  # number of test sample predicted on for the prediction on a horizon
sample_train = 150  # number of training sample points used by a trainable model

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
def real_predictions_new(file_pred: str, save_path: str):
    file_true = "../res/True_hosp_values_BE.csv"
    prediction_df = pd.read_csv(file_pred)
    true_df = pd.read_csv(file_true)

    prediction_last_walk = prediction_df[["DATE", "NEW_HOSP(t+1)"]]
    prediction_last_walk = prediction_last_walk[prediction_last_walk["DATE"].between("2021-03-01", "2021-03-31", inclusive=True)]

    true_forecast = true_df[true_df["DATE"].between("2021-03-02", "2021-04-01", inclusive=True)]
    true_forecast = true_forecast[["DATE", "NEW_HOSP"]]

    prediction_last_walk["DATE"] = true_forecast["DATE"].values
    prediction_last_walk = prediction_last_walk.set_index("DATE")
    true_forecast = true_forecast.set_index("DATE")

    samples = true_df[true_df["DATE"].between("2021-02-01", "2021-03-01", inclusive=True)]
    samples = samples[["DATE", "NEW_HOSP"]].set_index("DATE")

    print(true_forecast)

    # Plot
    fig = plt.figure(figsize=(5, 4))
    plt.plot(samples, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(true_forecast, linestyle='', marker='o', color=color_actual, label=f'True value t+1')
    plt.plot(prediction_last_walk, linestyle='', marker='X', color=color_prediction, label=f'Prediction t+1')
    ax = fig.axes[0]
    # set locator
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    # set formatter
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt_finish()
    plt.savefig(save_path, dpi=200)


def real_predictions_tot(file_pred: str, save_path: str):
    file_true = "../res/True_hosp_values_BE.csv"
    prediction_df = pd.read_csv(file_pred)
    true_df = pd.read_csv(file_true)

    basic_tot = true_df[true_df["DATE"].between("2021-03-01", "2021-03-31", inclusive=True)]
    basic_tot = basic_tot[["DATE", "TOT_HOSP"]].set_index("DATE")

    prediction_last_walk = prediction_df[["DATE", "NEW_HOSP(t+1)", "NEW_HOSP(t+2)", "NEW_HOSP(t+3)", "NEW_HOSP(t+4)", "NEW_HOSP(t+5)", "NEW_HOSP(t+6)", "NEW_HOSP(t+7)", "NEW_HOSP(t+8)", "NEW_HOSP(t+9)", "NEW_HOSP(t+10)"]]
    prediction_last_walk = prediction_last_walk[
        prediction_last_walk["DATE"].between("2021-03-01", "2021-03-31", inclusive=True)]

    true_forecast = true_df[true_df["DATE"].between("2021-03-11", "2021-04-10", inclusive=True)]
    true_forecast = true_forecast[["DATE", "TOT_HOSP"]]

    prediction_last_walk["DATE"] = true_forecast["DATE"].values
    prediction_last_walk = prediction_last_walk.set_index("DATE")
    true_forecast = true_forecast.set_index("DATE")

    total_pred = []
    for index, elem in enumerate(prediction_last_walk["NEW_HOSP(t+10)"]):
        final_sum = 0
        for i in range(1, 10):
            name = f"NEW_HOSP(t+{i})"
            final_sum += prediction_last_walk[name].values[index]
        final_sum += elem + basic_tot["TOT_HOSP"].values[index]
        total_pred.append(final_sum)

    prediction_last_walk["TOT_HOSP(t+10)"] = total_pred
    prediction_last_walk = prediction_last_walk.drop(columns=["NEW_HOSP(t+1)", "NEW_HOSP(t+2)", "NEW_HOSP(t+3)", "NEW_HOSP(t+4)", "NEW_HOSP(t+5)", "NEW_HOSP(t+6)", "NEW_HOSP(t+7)", "NEW_HOSP(t+8)", "NEW_HOSP(t+9)", "NEW_HOSP(t+10)"])

    samples = true_df[true_df["DATE"].between("2021-02-01", "2021-03-10", inclusive=True)]
    samples = samples[["DATE", "TOT_HOSP"]].set_index("DATE")

    # Plot
    fig = plt.figure(figsize=(5, 4))
    plt.plot(samples, linestyle='-', marker='o', color=color_train, label='Value of the last days')
    plt.plot(true_forecast, linestyle='', marker='o', color=color_actual, label=f'True value t+10')
    plt.plot(prediction_last_walk, linestyle='', marker='X', color=color_prediction, label=f'Prediction t+10')
    ax = fig.axes[0]
    # set locator
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    # set formatter
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt_finish()
    plt.savefig(save_path, dpi=200)


def plot_assembly_real_prediction(name_assembly_file: str, date: str):
    threshold = 0.3
    pred_file_init = f'{dir_results}/{name_assembly_file}_prediction_init_BE.csv'
    pred_file_assembly = f'{dir_results}/{name_assembly_file}_prediction_BE.csv'
    pred_file_c = f'{dir_results}/{name_assembly_file}_prediction_c_BE.csv'
    try:
        pred_df_init = pd.read_csv(pred_file_init)
        pred_df_assembly = pd.read_csv(pred_file_assembly)
        pred_df_c = pd.read_csv(pred_file_c)
    except FileNotFoundError:
        print(f'assembly file {name_assembly_file} not found')
        return
    columns = [i for i in pred_df_init.columns if '(t+' in i]
    columns_c = [i for i in pred_df_c.columns if '(t+' in i]
    begin = datetime.strptime(date, '%Y-%m-%d')
    date_range = pd.date_range(begin, begin + timedelta(days=len(columns)-1))
    pred_df_init = pred_df_init[pred_df_init['DATE'] == date][columns]
    pred_df_assembly = pred_df_assembly[pred_df_assembly['DATE'] == date][columns]
    pred_df_c = pred_df_c[pred_df_c['DATE'] == date][columns_c]

    url_world = "../data/hospi/world.csv"
    url_pop = "../data/population.txt"
    url_trends = "../data/trends/model/"
    url_hospi_belgium = "../data/hospi/be-covid-hospi.csv"
    url_department_france = "france_departements.csv"
    url_hospi_france_new = "../data/hospi/fr-covid-hospi.csv"
    url_hospi_france_tot = "../data/hospi/fr-covid-hospi-total.csv"
    df_hospi = util.hospi_french_region_and_be(url_hospi_france_tot, url_hospi_france_new, url_hospi_belgium,
                                           url_department_france, util.french_region_and_be, new_hosp=True,
                                           tot_hosp=True)['BE']
    df_hospi = df_hospi.reset_index().set_index('DATE')
    df_hospi = df_hospi.rolling(7, center=True).mean().dropna()
    target = df_hospi.loc[date_range]['NEW_HOSP']

    val = pred_df_init.values.reshape((-1, 1))
    pred_init = pd.DataFrame(data=val, index=date_range)
    val = pred_df_c.values.reshape((-1, 1))
    pred_c = pd.DataFrame(data=val, index=date_range)
    val = pred_df_assembly.values.reshape((-1, 1))
    pred_assembly = pd.DataFrame(data=val, index=date_range)

    fig = plt.figure(figsize=(5, 4))
    difference = pred_assembly - pred_init
    uplims = [j[0] > 0 for i, j in pred_init.iterrows()]
    lolims = [not i for i in uplims]
    difference = abs(difference).values.reshape(len(columns))
    x = pred_init.index
    pred_init = pred_init.values.reshape(len(columns))
    pred_assembly = pred_assembly.values.reshape(len(columns))
    plt.plot(x, pred_init, linestyle='-', marker='o', color=color_train, label='Initial prediction')
    plt.plot(x, pred_assembly, linestyle='', marker='X', color=color_prediction, label=f'Corrected prediction')
    plt.plot(target, linestyle='', marker='o', color=color_actual, label=f'True value')
    ax = fig.axes[0]
    ax.fill_between(x, pred_init - difference, pred_init + difference, alpha=0.2, interpolate=True, label='Correction range')
    # set locator
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt_finish()
    plt.savefig(f'../plot/predictions/prediction_assembler_behavior', dpi=200)


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
                pytrends.get_interest_over_time(search, cat=0, timeframe=random_timeframe(), geo=loc)
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

def plot_trends(df_plot, topic_code, show=True, figsize=None):
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if isinstance(df_plot, list):
        list_df = df_plot
    else:
        list_df = [df_plot]
    for df in list_df:
        df_plot = 100 * df[[topic_code]] / df[[topic_code]].max()
        plt.plot(df_plot)
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


def plot_collection_methods():
    # generate the model data using the different methods
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topic = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    begin = datetime.strptime('2020-02-01', trends_query.day_format)
    end = datetime.strptime('2021-02-01', trends_query.day_format)
    for method in ['daily', 'hourly', 'minimal']:
        df = trends_query.generate_model_data(method, topic, geo, savefile=False)[geo_code][topic_name]
        df = df[begin:end]
        plot_trends(df, topic_code, show=False)
        plt.tight_layout()
        plt.savefig(f'{plot_example_dir}/{method}_collection', dpi=200)
    df = pd.read_csv(f'../data/trends/samples/{geo_code}-{topic_name}_yearly.csv', parse_dates=['date'],
                                         date_parser=trends_query.date_parser_daily).set_index('date')
    df = df[begin:end]
    plot_trends(df, topic_code, show=False)
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/yearly_collection', dpi=200)


def plot_hourly_collection():
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topic = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    begin = datetime.strptime('2020-02-01', trends_query.day_format)
    end = datetime.strptime('2021-02-01', trends_query.day_format)
    dirpath = trends_query.dir_hourly
    filename = f'{geo_code}-{topic_name}.csv'
    df = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                         date_parser=trends_query.date_parser_hourly).set_index('date')
    df = df[begin:end]
    batches = sorted(df['batch_id'].unique(), key=lambda x: abs(x))
    list_batch = []
    for i in batches:
        df_batch = df[df['batch_id'] == i]
        list_batch.append(df_batch)
    plot_trends(list_batch, topic_code, show=False, figsize=(6,3))
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/hourly_method_batches', dpi=200)

    list_df_hourly = trends_query.ModelData.scale_df(df, topic_code)  # scale the dataframe
    plot_trends(list_df_hourly, topic_code, show=False, figsize=(6,3))
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/hourly_method_scaled', dpi=200)

    list_df_hourly = trends_query.HourlyModelData.drop_incomplete_days(
        list_df_hourly)  # drop the incomplete days (check doc for details)
    list_df_hourly = [df.resample('D').mean() for df in list_df_hourly]
    plot_trends(list_df_hourly, topic_code, show=False, figsize=(6,3))
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/hourly_method_resampled', dpi=200)


def plot_mean_daily():
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topic = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    filename = f'{geo_code}-{topic_name}_daily.csv'
    dirpath = '../data/trends/samples'
    df = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                         date_parser=trends_query.date_parser_daily).set_index('date')
    val_min = np.ones(len(df)) * 100
    val_max = np.zeros(len(df))
    for column in df.columns:
        if column != topic_code and column != 'batch_id':
            val_min = np.minimum(val_min, df[column].array)
            val_max = np.maximum(val_max, df[column].array)
    val_min = pd.DataFrame(data=val_min, index=df.index)
    val_max = pd.DataFrame(data=val_max, index=df.index)
    mean = df[topic_code]
    fig = plt.figure()
    plt.plot(mean, label='Mean of 100 queries')
    plt.plot(val_min, linestyle=':', color='r', label='Extremum between all queries')
    plt.plot(val_max, linestyle=':', color='r')
    ax = fig.axes[0]
    # set monthly locator
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.legend()
    plt.savefig(f'{plot_example_dir}/error_margin_daily', dpi=200)
    # plot for convergence of error
    list_df = []
    for i in range(100):
        column = [f'{topic_code}_{j}' for j in range(i)]
        list_df.append(df[column].mean(axis=1))
    error_list = []
    for df_a, df_b in zip(list_df, list_df[1:]):
        error_list.append(np.mean(abs(df_a - df_b)))
    plt.figure(figsize=(4,3))
    plt.plot(range(1, 100), error_list)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/mean_daily', dpi=200)


def plot_scale_df():
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topic = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    dirpath = '../data/trends/samples'
    for method, date_parser in [('daily', trends_query.date_parser_daily), ('hourly', trends_query.date_parser_hourly)]:
        filename = f'{geo_code}-{topic_name}_{method}_true.csv'
        filename_1 = f'{geo_code}-{topic_name}_{method}_1.csv'
        filename_2 = f'{geo_code}-{topic_name}_{method}_2.csv'

        df = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                             date_parser=date_parser).set_index('date')
        df_1 = pd.read_csv(f'{dirpath}/{filename_1}', parse_dates=['date'],
                                             date_parser=date_parser).set_index('date')
        df_2 = pd.read_csv(f'{dirpath}/{filename_2}', parse_dates=['date'],
                                             date_parser=date_parser).set_index('date')
        df = df[[topic_code]]
        df_1 = df_1[[topic_code]]
        df_2 = df_2[[topic_code]]
        fig = plt.figure()
        plt.plot(df_1)
        plt.plot(df_2)
        ax = fig.axes[0]
        # set monthly locator
        if method == 'daily':
            locator = mdates.MonthLocator(interval=1)
        else:
            locator = mdates.DayLocator(interval=1)
        ax.xaxis.set_major_locator(locator)
        # set formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        # set font and rotation for date tick labels
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{plot_example_dir}/trends_{method}_scaling_before', dpi=200)

        fig = plt.figure()
        df_scaled = trends_query.ModelData.merge_trends_batches(df_1, df_2, topic_code)
        plt.plot(df, label='true request')
        plt.plot(df_scaled, label='scaled request', linestyle='-.')
        plt.legend()
        ax = fig.axes[0]
        # set monthly locator
        ax.xaxis.set_major_locator(locator)
        # set formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        # set font and rotation for date tick labels
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{plot_example_dir}/trends_{method}_scaling_after', dpi=200)


def plot_adjacent_queries():
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topic = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    dirpath = '../data/trends/samples'
    filename_1 = f'{geo_code}-{topic_name}_daily_left.csv'
    filename_2 = f'{geo_code}-{topic_name}_daily_right.csv'
    date_parser = trends_query.date_parser_daily
    df_1 = pd.read_csv(f'{dirpath}/{filename_1}', parse_dates=['date'],
                       date_parser=date_parser).set_index('date')
    df_2 = pd.read_csv(f'{dirpath}/{filename_2}', parse_dates=['date'],
                       date_parser=date_parser).set_index('date')
    df_1 = df_1[[topic_code]]
    df_2 = df_2[[topic_code]]
    fig = plt.figure()
    plt.plot(df_1)
    plt.plot(df_2)
    ax = fig.axes[0]
    # set monthly locator
    locator = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(locator)
    # set formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_example_dir}/adjacent_queries', dpi=200)


if __name__ == '__main__':
    # tor_vs_local()
    # plot_tor_vs_local()
    # df = pd.read_csv('../data/trends/model/FR-B-Fièvre.csv', parse_dates=['date']).set_index('date')
    # plot_trends(df, df.columns[0])
    # plot_collection_methods()
    # plot_hourly_collection()
    # plot_mean_daily()
    # plot_scale_df()
    # plot_adjacent_queries()
    # plot_prediction_t_1()
    # plot_prediction()
    # plot_assembly_real_prediction('2021-06-02-11:26_get_assembly_NEW_HOSP', '2021-04-20')

    dict_models_france = {
        "../res/2021-06-02-15:49_get_custom_linear_regression_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_linear_reg_chap6', '../plot/predictions/prediction_linear_reg_tot'],
        "../res/2021-06-02-15:49_get_assembly_2_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_assembler_chap6', '../plot/predictions/prediction_assembler_tot'],
        "../res/2021-06-02-15:49_get_baseline_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_baseline_chap6', '../plot/predictions/prediction_baseline_tot'],
        "../res/2021-06-02-15:49_get_dense_model_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_dense_chap6', '../plot/predictions/prediction_dense_tot'],
        "../res/2021-06-02-15:49_get_encoder_decoder_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_encoder-decoder_chap6',
            '../plot/predictions/prediction_encoder-decoder_tot'],
        "../res/2021-06-05-11:27_get_encoder_decoder_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_encoder-decoder_no_trends_chap6',
            '../plot/predictions/prediction_encoder-decoder_no_trends_tot'],
    }

    dict_models_europe = {
        "../res/2021-06-03-00:19_get_custom_linear_regression_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_linear_reg_chap6_europe', '../plot/predictions/prediction_linear_reg_tot_europe'],
        "../res/2021-06-03-00:19_get_assembly_2_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_assembler_chap6_europe', '../plot/predictions/prediction_assembler_tot_europe'],
        "../res/2021-06-03-00:19_get_baseline_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_baseline_chap6_europe', '../plot/predictions/prediction_baseline_tot_europe'],
        "../res/2021-06-03-00:19_get_dense_model_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_dense_chap6_europe', '../plot/predictions/prediction_dense_tot_europe'],
        "../res/2021-06-03-00:19_get_encoder_decoder_NEW_HOSP_prediction_BE.csv": [
            '../plot/predictions/prediction_encoder-decoder_europe',
            '../plot/predictions/prediction_encoder-decoder_tot_europe'],
        # "../res/2021-06-03-00:19_get_encoder_decoder_NEW_HOSP_prediction_BE.csv": ['../plot/predictions/prediction_encoder-decoder_no_trends_chap6','../plot/predictions/prediction_encoder-decoder_no_trends_tot'],
    }

    for key, val in dict_models_france.items():
        real_predictions_new(key, val[0])
        real_predictions_tot(key, val[1])

    #file_pred = "../res/2021-06-03-00:19_get_assembly_2_NEW_HOSP_prediction_BE.csv"
    #real_predictions_tot(file_pred)
