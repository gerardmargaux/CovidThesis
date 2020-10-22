import scaler
from tensorflow import keras
import numpy as np
import talos
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM
from tensorflow.keras import regularizers
import pickle
from tensorflow.python.keras.models import load_model
try:
    from src.data_collection import *
except ModuleNotFoundError:
    from data_collection import *
import urllib

look_back = 1
delay = 7
use_full_valid_test = True
train_ratio = 0.50
valid_ratio = 0.25
test_ratio = 1.0 - train_ratio - valid_ratio
n_features = -1

train_data = {}
valid_data = {}
test_data = {}

list_topics = {
    'Fièvre': '/m/0cjf0',
    'Mal de gorge': '/m/0b76bty',
    'Dyspnée': '/m/01cdt5',
    'Agueusie': '/m/05sfr2',
    'Anosmie': '/m/0m7pl',
    'Coronavirus': '/m/01cpyy',
    'Virus': '/m/0g9pc',
    'Température corporelle humaine': '/g/1213j0cz',
    'Épidémie': '/m/0hn9s',
    'Symptôme': '/m/01b_06',
    'Thermomètre': '/m/07mf1',
    'Grippe espagnole': '/m/01c751',
    'Paracétamol': '/m/0lbt3',
    'Respiration': '/m/02gy9_',
    'Toux': '/m/01b_21'
}


def create_full_df(url_trends, url_hospi, geo):
    df_final = create_dataframe_belgium(url_hospi)
    for term in list_topics.keys():
        path = f"{url_trends}{geo}-{term}.csv"
        encoded_path = requests.get(path).content
        df_trends = pd.read_csv(io.StringIO(encoded_path.decode("utf-8"))).rename(columns={"date": "DATE"})
        df_trends['LOC'] = 'Belgique'
        df_trends.set_index(['LOC', 'DATE'], inplace=True)
        df_final = pd.concat([df_final, df_trends], axis=1)

    return df_final


def create_df_hospi(url):
    """
    Creates a dataframe containing the number of new hospitalizations with respect to the date
    :param url: url of the CSV file in which data concerning hospitalizations are stocked
    """
    df_hospi = pd.read_csv(url).groupby(["DATE"]).agg({"NEW_IN": "sum"}).reset_index().rename(columns={"NEW_IN": "HOSP"})
    return df_hospi


def create_df_trends_url(url_hospi, geo):
    """
    Creates a dataframe containing the number of new hospitalizations and the trends of some symptoms with respect to
    the date
    :param url_trends:  url of the CSV file in which data concerning trends are stocked
    :param url_hospi: url of the CSV file in which data concerning hospitalizations are stocked
    :param geo: geo localisation of the trends requests
    :return: a full dataframe that is ready to be process
    """
    df_final = create_dataframe_belgium(url_hospi)
    """for term in list_topics.keys():
        print(term)
        path = f"{url_trends}{geo}-{term}.csv"
        encoded_path = requests.get(path).content
        df_trends = pd.read_csv(io.StringIO(encoded_path.decode("utf-8"))).rename(columns={"date": "DATE"})
        df_trends['LOC'] = 'Belgique'
        df_trends.set_index(['LOC', 'DATE'], inplace=True)
        df_final = pd.concat([df_final, df_trends], axis=1)"""
    return df_final


def create_dataset(dataset, look_back=3):
    """
    Converts an array of values into a dataset matrix
    :param dataset: array of values
    :return: a dataset matrix
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - delay):
        a = dataset[i:(i + look_back), :-1]
        # print(a)
        dataX.append(a)
        dataY.append(dataset[i + look_back + delay - 1, -1])
    return np.array(dataX), np.array(dataY)


def create_datapoints_for_each_loc(start_year, start_mon, stop_year, stop_mon, step,
                                   data_collection=True):
    """
    Creates keras datasets (train, test and validation datapoints) for each location.
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :param step: number of iteration for finding new related topics
    :param data_collection: is true is we need to collect new data, false otherwise
    """
    global n_features
    global train_data
    global valid_data
    global test_data
    full_datapoints = {}

    geo = "BE"
    url_hospi = "https://raw.githubusercontent.com/gerardmargaux/CovidThesis/master/data/hospi/be-covid-hospi.csv"
    full_df = create_df_trends_url(url_hospi, geo)
    print(full_df)
    full_data, full_data_no_rolling = google_trends_process(full_df, list_topics, start_year=start_year, start_mon=start_mon,
                                                            stop_year=stop_year, stop_mon=stop_mon, step=step,
                                                            data_collection=data_collection)

    for loc in full_data.index.levels[0]:
        x, y = create_dataset(full_data.loc[loc].values)
        full_datapoints[loc] = (x, y)
        assert n_features == -1 or n_features == x.shape[-1]
        n_features = x.shape[-1]

    if not use_full_valid_test:
        for loc in full_datapoints:
            x, y = full_datapoints[loc]
            length = x.shape[0]
            train_len = int(length * train_ratio)
            valid_len = int(length * valid_ratio)

            train_data[loc] = (x[0:train_len], y[0:train_len])
            valid_data[loc] = (x[train_len:train_len + valid_len], y[train_len:train_len + valid_len])
            test_data[loc] = (x[train_len + valid_len:], y[train_len + valid_len:])

    else:
        all_locs = list(full_datapoints.keys())
        np.random.shuffle(all_locs)

        length = len(all_locs)
        train_len = int(length * train_ratio)
        valid_len = int(length * valid_ratio)

        train_data = {loc: full_datapoints[loc] for loc in all_locs[0:train_len]}
        valid_data = {loc: full_datapoints[loc] for loc in all_locs[train_len:train_len + valid_len]}
        test_data = {loc: full_datapoints[loc] for loc in all_locs[train_len + valid_len:]}


def create_toy_model():
    """
    Creates a toy model
    :return: the model created
    """
    toy_model = Sequential()
    toy_model.add(LSTM(32, return_sequences=True, input_shape=(None, n_features)))
    toy_model.add(LSTM(1, return_sequences=True))
    toy_model.add(TimeDistributed(Dense(1)))
    toy_model.compile(loss="mse", optimizer='adam')
    toy_history = toy_model.fit(train_generator(), steps_per_epoch=len(train_data), epochs=400, verbose=1, shuffle=False,
                                validation_data=validation_generator(), validation_steps=len(valid_data))
    show_datapoints(toy_model)
    return toy_model, toy_history


def train_generator():
    """
    Creates a generator for the datapoints used for the training
    :return: a generator containing the datapoints for the training
    """
    while True:
        for loc in train_data:
            # let's remove some "0" points, just to ensure the LSTM does not directly remember
            # where the peaks are
            d = np.random.randint(0, 20)
            yield train_data[loc][0][d:], train_data[loc][1][d:]


def validation_generator():
    """
    Creates a generator for the datapoints used for the validation
    :return: a generator containing the datapoints for the validation
    """
    while True:
        for loc in valid_data:
            yield valid_data[loc]


def show_datapoints(model):
    """
    Shows the distribution of the datapoints in a plot
    :param model: the model used for the prediction
    """
    for loc in train_data:
        print("TRAINING", loc)
        x, y = train_data[loc]
        yp = model.predict(x)

        plt.plot(y)
        plt.plot(yp.reshape(-1))
        plt.show()

    for loc in valid_data:
        print("VALIDATION", loc)
        x, y = valid_data[loc]
        yp = model.predict(x)

        plt.plot(y)
        plt.plot(yp.reshape(-1))
        plt.show()

    for loc in test_data:
        print("TEST", loc)
        x, y = test_data[loc]
        yp = model.predict(x)

        plt.plot(y)
        plt.plot(yp.reshape(-1))
        plt.show()


def create_model(_, _2, _3, _4, p):
    """
    Trains the sequential model with all the train_datapoints and saves this model.
    :param _: X training datapoints
    :param _2: Y training datapoints
    :param _3: X validation datapoints
    :param _4: Y validation datapoints
    :param p: hyper parameters to evaluate
    :return: history : a history object containing a dictionary of all loss values and other metric values.
    :return: model : the sequential trained model
    """
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    model = Sequential()
    model.add(LSTM(p["n_lstm_node_first"], return_sequences=True, input_shape=(None, n_features),
                   kernel_regularizer=p['reg'](p['regw'])))
    if p["n_lstm_node_second"] != 0:
        model.add(LSTM(p["n_lstm_node_second"], return_sequences=True, kernel_regularizer=p['reg'](p['regw'])))
    for _ in range(p["n_layers_after"]):
        model.add(TimeDistributed(Dense(p["n_node_hidden_layers"], kernel_regularizer=p['reg'](p['regw']),
                                        activation=p['activation'])))
    model.add(TimeDistributed(Dense(1, kernel_regularizer=p['reg'](p['regw']))))

    model.compile(loss=p["losses"], optimizer=p["optimizer"], metrics=['mae', 'mse'])

    # With TensorFlow version 2.2 or higher, the fit function works exactly as the fit_generator function
    # Not everything is stocked into the RAM
    history = model.fit(train_generator(), steps_per_epoch=len(train_data), epochs=p["epochs"], verbose=0, shuffle=False,
                        validation_data=validation_generator(), validation_steps=len(valid_data))

    # saving of the history in a log file
    #with open('../data/trends/training.log', 'wb') as file_pi:
    #    pickle.dump(history.history, file_pi)

    # saving of the entire model
    # path to the file where the model will be saved
    #save_model = "../data/trends/saved_model"
    #tf.keras.models.save_model(model=model, filepath=save_model)

    return history, model


def loaded_model(_, _2, _3, _4, p):
    """
    Loads the sequential model saved previously.
    :param _: X training datapoints
    :param _2: Y training datapoints
    :param _3: X validation datapoints
    :param _4: Y validation datapoints
    :param p: hyper parameters to evaluate
    :return: history : a history object containing a dictionary of all loss values and other metric values.
    :return: model : the sequential trained model
    """
    # path to the file where the model will be saved
    save_model = "../data/trends/saved_model"

    # Creation and fitting of the toy model
    toy_model, history = create_toy_model()

    model = load_model(save_model)
    # Loading of the history in a log file
    history.history = pickle.load(open('../data/trends/training.log', "rb"))

    return history, model


def define_parameters():
    """
    Defines the parameters used in order to find the model that fits the training and validation datapoints best.
    :return: the parameters given to the scan object
    """
    p = {'activation': ['relu', 'elu', 'sigmoid'],
         'n_layers_after': [0, 1, 2],
         'n_node_hidden_layers': [10, 30, 50],
         'n_lstm_node_first': [10, 20, 30],
         'n_lstm_node_second': [0, 10, 20, 30],
         'reg': [lambda x: regularizers.l2(l=x), lambda x: regularizers.l1(l=x), lambda x: None],
         'regw': [1e-4, 5e-4, 1e-3],
         'optimizer': ['Adam', 'sgd'],
         'losses': ['mae', 'mse'],
         'epochs': [300, 500],
         }

    return p


def find_best_model(loading=True):
    """
    Finds the best model with respect to the value of the MSE
    :param loading: is true if we want to load a previously saved model
    :return: the best model found
    """
    p = define_parameters()

    # If loading is false, we need to train the entire model
    if not loading:
        scan_object = talos.Scan(
            x=[],
            y=[],
            x_val=[],
            y_val=[],
            params=p,
            model=create_model,
            experiment_name='trends1',
            fraction_limit=0.01
        )

    # If loading is true a model is already saved -> we load it
    else:
        scan_object = talos.Scan(
            x=[],
            y=[],
            x_val=[],
            y_val=[],
            params=p,
            model=loaded_model,
            experiment_name='trends1',
            fraction_limit=0.01
        )

    analyze_object = talos.Analyze(scan_object)
    print("MAE", analyze_object.low('mae'))
    print("MSE", analyze_object.low('mse'))
    print("VAL MAE", analyze_object.low('val_mae'))
    print("VAL MSE", analyze_object.low('val_mse'))
    analyze_object.table('val_mse', exclude=[], ascending=True)

    # Finding the model that minimises the value of the MSE
    best_model = scan_object.best_model('val_mse', asc=True)

    return best_model


def prediction_model(model):
    """
    Predicts the test datapoints for a set of localisation and computes the error of the model based on the MSE and MAE.
    :param model: model that fits the training and validation datapoints best
    """
    for name, datapoints in [("Train", train_data), ("Val.", valid_data), ("Test", test_data)]:
        print(f"\\midrule {name}")
        total_mse = 0.0
        total_mae = 0.0
        for loc in datapoints:
            pred = model.predict(datapoints[loc][0])
            pred = pred.reshape(-1)
            error_mse = ((pred - datapoints[loc][1]) ** 2).mean()
            error_mae = (np.absolute(pred - datapoints[loc][1])).mean()
            print(f"& {loc} & {error_mse * 1000:.2f}e-3 & {error_mae:.3f} \\\\")
            total_mse += error_mse
            total_mae += error_mae

        total_mse /= len(datapoints)
        total_mae /= len(datapoints)
        print(f"& \\textbf{{Overall}} & {total_mse * 1000:.2f}e-3 & {total_mae:.3f} \\\\")

    f, (axes) = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(7.5, 9))

    idx = 0
    for name, datapoints in [("Val.", valid_data), ("Test", test_data)]:
        for loc in datapoints:

            print(name, loc)
            pred = model.predict(datapoints[loc][0])
            pred = pred.reshape(-1)

            axes[int(idx / 2)][int(idx % 2)].set_title(loc)
            axes[int(idx / 2)][int(idx % 2)].plot(datapoints[loc][1])
            axes[int(idx / 2)][int(idx % 2)].plot(pred)

            idx += 1

            if idx == 1:
                axes[int(idx / 2)][int(idx % 2)].axis("off")
                idx += 1
    axes[-1][0].set_xlabel('Days since 1st February')
    axes[-1][1].set_xlabel('Days since 1st February')
    f.show()
    f.tight_layout()
    f.savefig('results.pdf')


def run_automatically(start_year, start_mon, stop_year, stop_mon, step, data_collection=True):
    """
    Runs all the code automatically : data collection + prediction model
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :param step: number of iteration for finding new related topics
    :param data_collection: is true is we need to collect new data, false otherwise
    """
    loading = True
    if data_collection:
        loading = False

    create_datapoints_for_each_loc(start_year=start_year, start_mon=start_mon, stop_year=stop_year,
                                   stop_mon=stop_mon, step=step, data_collection=data_collection)

    # Creation and fitting of the keras model
    best_model = find_best_model(loading=loading)
    prediction_model(best_model)


if __name__ == "__main__":
    # For collecting data before creating the prediction model -> data_collection=True
    # run_automatically(start_year=2020, start_mon=2, stop_year=2020, stop_mon=9, step=1, data_collection=True)
    geo = "BE"
    url_trends = "https://raw.githubusercontent.com/gerardmargaux/CovidThesis/master/data/trends/model/"
    url_hospi = "https://raw.githubusercontent.com/gerardmargaux/CovidThesis/master/src/be-covid-hospi.csv"
    full_df = create_full_df(url_trends, url_hospi, geo)
