#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
from tensorflow import keras
from pytrends.dailydata import get_daily_data
import os.path
import numpy as np
import talos
import matplotlib.pyplot as plt
from pandas import read_csv, Series
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, TimeDistributed, LSTM
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta
import re
import pickle

from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.models import model_from_json, load_model

np.random.seed(7)

terms = {
    "symptomes_keyword": "/m/01b_06",
    "toux_sujet": "/m/01b_21",
}

simple_terms = [

]

google_geocodes = {
    'FR-A': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-B': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-C': "Auvergne-Rhône-Alpes",
    'FR-P': "Normandie",
    'FR-D': "Bourgogne-Franche-Comté",
    'FR-E': 'Bretagne',
    'FR-F': 'Centre-Val de Loire',
    'FR-G': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-H': 'Corse',
    'FR-I': "Bourgogne-Franche-Comté",
    'FR-Q': "Normandie",
    'FR-J': 'Ile-de-France',
    'FR-K': 'Languedoc-Roussillon-Midi-Pyrénées',
    'FR-L': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-M': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-N': 'Languedoc-Roussillon-Midi-Pyrénées',
    'FR-O': 'Nord-Pas-de-Calais-Picardie',
    'FR-R': 'Pays de la Loire',
    'FR-S': 'Nord-Pas-de-Calais-Picardie',
    'FR-T': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-U': "Provence-Alpes-Côte d'Azur",
    'FR-V': "Auvergne-Rhône-Alpes",
    'BE': "Belgique"
}

terms.update({x: x for x in simple_terms})

# In[76]:


# shameless copy of pytrends

from datetime import date, timedelta
from functools import partial
from time import sleep
from calendar import monthrange
import pandas as pd
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq


def extract_topics(filename="topics.txt", toList=False):
    """
    Extracts the pairs of "topics_mid topic_title" in the file provided
    :param filename: file were the topics are written. Can be both a filename or a list of files
        Each valid line must be in the format "topic_mid topic_title"
    :param toList: whether to return the list of keys or the whole dictionary
    :return: dictionary of {topic_title: topic_mid} for each topic provided
    """
    results = {}
    pattern = "(.+)\s(/m/.+)"
    if isinstance(filename, str):
        filename2 = [filename]
    else:
        filename2 = filename
    for name in filename2:
        with open(name) as file:
            for line in file:
                search_obj = re.match(pattern, line)
                if search_obj is not None:
                    results[search_obj.group(1)] = search_obj.group(2)
    return results if not toList else list(results.keys())


def compute_corr(left, delta, full_data_be, init_df):
    thisdata = pd.concat([init_df[left].shift(delta), full_data_be["HOSP"]], axis=1)
    return thisdata.corr()["HOSP"][left]


def relevant_pytrends(init_file,
                      start_year: int,
                      start_mon: int,
                      stop_year: int,
                      stop_mon: int,
                      geo: str = 'BE',
                      verbose: bool = True,
                      wait_time: float = 5.0,
                      step=0,
                      threshold=0.65):
    """
    get the relevant google trends
    :param init_file: init file of keywords
    :param step: number of iteration. 0 -> return df for the init_file topics/queries only
    :return:
    """
    start_date = date(start_year, start_mon, 1)
    stop_date = get_last_date_of_month(stop_year, stop_mon)
    init_topics = extract_topics()  # initial dict of topics
    init_df = pd.concat([load_term(key, val) for key, val in init_topics.items()], axis=1)  # interest over time for each topics
    init_df = init_df.rolling(7, center=True).mean().dropna()  # rolling average
    data_be = pd.read_csv("be-covid-hospi.csv").groupby(["DATE"]).agg({"NEW_IN": "sum"}).rename(
        columns={"NEW_IN": "HOSP"})  # data for belgium
    data_be = data_be.rolling(7, center=True).mean().dropna()

    # Add "fake" data (zeroes before the beginning of the crisis) for each loc
    toadd = []
    min_index = data_be.index.min()
    end = datetime.strptime(min_index, "%Y-%m-%d").date()
    min_index_init_df = init_df.index.min()
    cur = datetime.strptime(min_index_init_df, "%Y-%m-%d").date()

    while cur != end:
        toadd.append([cur.strftime("%Y-%m-%d"), 0])
        cur += timedelta(days=1)
    data_be = data_be.reset_index().append(pd.DataFrame(toadd, columns=["DATE", "HOSP"])).set_index("DATE")

    out = []
    for key, val in init_topics.items():
        correlations = [(delay, compute_corr(key, delay, data_be, init_df)) for delay in range(0, 17)]
        best_delay, best_corr = max(correlations, key=lambda x: abs(x[1]))
        out.append((key, val, best_delay, best_corr))
    out = list(filter(lambda x: abs(x[3]) > threshold, out))  # keep only the most correlated topics
    out = [x[0] for x in out]  # keep only the term, not the tuples

    init_df = init_df[out]  # filter the df
    df = pd.DataFrame(columns=("Topic_title", "Topic_mid", "Best delay", "Best correlation"))

    for i in range(step):
        total_topics = {}
        for term in out:
            pytrends = TrendReq(hl='fr-BE')
            # Initialize build_payload with the word we need data for
            build_payload = partial(pytrends.build_payload,
                                    kw_list=[term], cat=0, geo=geo, gprop='')
            # new topics to evaluate
            new_topics = _fetch_related(pytrends, build_payload, convert_dates_to_timeframe(start_date, stop_date), term)
            total_topics = {**new_topics, **total_topics}

        # evaluate new topics over time
        list_topics = [load_term(key, val) for key, val in total_topics.items()]
        if list_topics:
            related_df = pd.concat(list_topics, axis=1)  # interest over time for each topics
            related_df = related_df.rolling(7, center=True).mean().dropna()  # rolling average

        # keep the most relevant topics
        out = []
        for key, val in total_topics.items():
            correlations = [(delay, compute_corr(key, delay, data_be, related_df)) for delay in range(0, 17)]
            best_delay, best_corr = max(correlations, key=lambda x: abs(x[1]))
            out.append([key, val, best_delay, best_corr])

        j = len(df)
        for topics in out:
            df.loc[j] = topics
            j += 1

        out = list(filter(lambda x: abs(x[3]) > threshold, out))  # keep only the most correlated topics
        out = [x[0] for x in out]  # keep only the term, not the tuples

        # if out is empty -> no new topic with a sufficient threshold is added to init_df
        if out:
            related_df = related_df[out]
            init_df = pd.concat([init_df, related_df], axis=1, join='inner')  # append new topics
            init_df = init_df.loc[:, ~init_df.columns.duplicated()]

    df.drop_duplicates(subset='Topic_title', keep='first', inplace=True)
    df.reset_index(0, inplace=True, drop=True)
    df.to_csv('../data/trends/explore/related_topics.csv')


def get_best_topics(filename='../data/trends/explore/related_topics.csv', number=15):
    """
    Gets the first best topics sorted with respect to the correlation.
    :param filename: path to the file where all the topics generated are stocked
    :param number: number of best topics that we keep
    :return: a dictionary containing topic_title:topic_mid pairs
    """
    df = pd.read_csv(filename)
    df['Best correlation'] = df['Best correlation'].abs()
    df.sort_values('Best correlation', ascending=False, inplace=True)
    df = df[['Topic_title', 'Topic_mid']]
    list_topics = df.to_dict('split')['data']

    dict = {}
    for i in range(min(number, len(list_topics))):
        dict[list_topics[i][0]] = list_topics[i][1]

    return dict


def get_last_date_of_month(year: int, month: int) -> date:
    """Given a year and a month returns an instance of the date class
    containing the last day of the corresponding month.
    Source: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
    """
    return date(year, month, monthrange(year, month)[1])


def convert_dates_to_timeframe(start: date, stop: date) -> str:
    """Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """Attempts to fecth data and retries in case of a ResponseError."""
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    return pytrends.interest_over_time()


def _fetch_related(pytrends, build_payload, timeframe: str, term) -> dict:
    """ Attempts to get the related topics of a particular term and retries in case of ResponseError"""
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    df = pytrends.related_topics()

    if df[term]['rising'].empty:
        return {}

    dic_rising = df[term]['rising'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()['topic_mid']
    dic_top = df[term]['top'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()['topic_mid']
    return {**dic_rising, **dic_top}  # return dic of related topics


def get_daily_data(word: str,
                   start_year: int,
                   start_mon: int,
                   stop_year: int,
                   stop_mon: int,
                   geo: str = 'BE',
                   verbose: bool = True,
                   wait_time: float = 5.0) -> pd.DataFrame:
    """Given a word, fetches daily search volume data from Google Trends and
    returns results in a pandas DataFrame.
    Details: Due to the way Google Trends scales and returns data, special
    care needs to be taken to make the daily data comparable over different
    months. To do that, we download daily data on a month by month basis,
    and also monthly data. The monthly data is downloaded in one go, so that
    the monthly values are comparable amongst themselves and can be used to
    scale the daily data. The daily data is scaled by multiplying the daily
    value by the monthly search volume divided by 100.
    For a more detailed explanation see http://bit.ly/trendsscaling
    Args:
        word (str): Word to fetch daily data for.
        start_year (int): the start year
        start_mon (int): start 1st day of the month
        stop_year (int): the end year
        stop_mon (int): end at the last day of the month
        geo (str): geolocation
        verbose (bool): If True, then prints the word and current time frame
            we are fecthing the data for.
    Returns:
        complete (pd.DataFrame): Contains 4 columns.
            The column named after the word argument contains the daily search
            volume already scaled and comparable through time.
            The column f'{word}_unscaled' is the original daily data fetched
            month by month, and it is not comparable across different months
            (but is comparable within a month).
            The column f'{word}_monthly' contains the original monthly data
            fetched at once. The values in this column have been backfilled
            so that there are no NaN present.
            The column 'scale' contains the scale used to obtain the scaled
            daily data.
    """

    # Set up start and stop dates
    start_date = date(start_year, start_mon, 1)
    stop_date = get_last_date_of_month(stop_year, stop_mon)

    # Start pytrends for BE region
    pytrends = TrendReq(hl='fr-BE')
    # Initialize build_payload with the word we need data for
    build_payload = partial(pytrends.build_payload,
                            kw_list=[word], cat=0, geo=geo, gprop='')

    if (stop_date - start_date).days >= 365:
        # Obtain monthly data for all months in years [start_year, stop_year]
        monthly = _fetch_data(pytrends, build_payload,
                                      convert_dates_to_timeframe(start_date, stop_date))

        # Get daily data, month by month
        results = {}
        # if a timeout or too many requests error occur we need to adjust wait time
        current = start_date
        while current < stop_date:
            last_date_of_month = get_last_date_of_month(current.year, current.month)
            timeframe = convert_dates_to_timeframe(current, last_date_of_month)
            if verbose:
                print(f'{word}:{timeframe}')
            results[current] = _fetch_data(pytrends, build_payload, timeframe)
            current = last_date_of_month + timedelta(days=1)
            sleep(wait_time)  # don't go too fast or Google will send 429s

        daily = pd.concat(results.values()).drop(columns=['isPartial'])
        complete = daily.join(monthly, lsuffix='_unscaled', rsuffix='_monthly')

        # Scale daily data by monthly weights so the data is comparable
        complete[f'{word}_monthly'].ffill(inplace=True)  # fill NaN values
        complete['scale'] = complete[f'{word}_monthly'] / 100
        complete[word] = complete[f'{word}_unscaled'] * complete.scale

    else:
        complete = _fetch_data(pytrends, build_payload,
                                      convert_dates_to_timeframe(start_date, stop_date))

    """ dataframe contains
    - word_unscaled: data with a top 0-100 monthly
    - word_monthly: data with a top 0-100 for start_date to end_date
    - scale: word_monthly/100
    - word: word_unscaled * scale -> contains data between 0-100 (with only one 100)
    """

    return complete


# Own code
def _dl_term(term, geo="BE-WAL", start_year=2020, start_mon=2, stop_year=2020, stop_mon=9):
    df = get_daily_data(term, start_year=start_year, start_mon=start_mon, stop_year=stop_year, stop_mon=stop_mon, geo=geo, verbose=False)
    if df.empty:
        return df
    else:
        return df[term].copy()


def load_term(termname, term,  dir="../data/trends/explore/", geo="BE-WAL", start_year=2020, start_mon=2, stop_year=2020, stop_mon=9):
    if "/" in termname:
        termname = termname.replace("/", "-")

    path = f"{dir}{geo}-{termname}.csv"

    if not os.path.exists(path):
        print(f"DL {geo} {termname}")
        content = _dl_term(term, start_year=2020, start_mon=2, stop_year=2020, stop_mon=9)
        content.to_csv(path)

    content = pd.read_csv(path)
    content = content.rename(columns={term: termname})
    content = content.set_index("date")
    return content


# French data
departements = pd.read_csv('france_departements.csv')
hospitalisations = pd.read_csv('hospitals.csv', sep=";")
data_fr = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(["regionName", "jour"],
                                                                                            as_index=False).agg(
    {"incid_hosp": "sum"})
data_fr = data_fr.rename(columns={"jour": "DATE", "incid_hosp": "HOSP", "regionName": "LOC"})

# Belgian data
data_be = pd.read_csv("be-covid-hospi.csv").groupby(["DATE"]).agg({"NEW_IN": "sum"}).reset_index().rename(
    columns={"NEW_IN": "HOSP"})
data_be["LOC"] = "Belgique"

# Full data
full_data = data_fr.append(data_be).set_index(["LOC", "DATE"])

# find smallest date for each loc
smallest = {}
for loc, date_current in full_data.index:
    if loc not in smallest or smallest[loc] > date_current:
        smallest[loc] = date_current

base_date = datetime.strptime("2020-02-01", "%Y-%m-%d").date()

# Add "fake" data (zeroes before the beginning of the crisis) for each loc
toadd = []
for loc, sm in smallest.items():
    end = datetime.strptime(sm, "%Y-%m-%d").date()
    cur = base_date

    while cur != end:
        toadd.append([cur.strftime("%Y-%m-%d"), loc, 0])
        cur += timedelta(days=1)

full_data = full_data.reset_index().append(pd.DataFrame(toadd, columns=["DATE", "LOC", "HOSP"])).set_index(
    ["LOC", "DATE"])


def normalize_hosp_stand(full_data):
    """
    Normalizes and standardizes the number of new hospitalizations between 0 and 1 PER LOC.
    The goal is to predict peaks/modification of the slope, not numbers.
    """
    full_data = full_data.reset_index()
    full_data["HOSP_CORR"] = full_data.groupby(["LOC"])['HOSP'].transform(lambda x: x / max(x))

    for x in full_data.columns:
        if x not in ["HOSP", "HOSP_CORR", "LOC", "DATE"]:
            # Standardisation between -1 and 1
            full_data[x] = (full_data.groupby(["LOC"])[x].transform(lambda x: x / max(x)) * 2.0) - 1.0

    # Set the final index and sort it
    full_data = full_data.set_index(["LOC", "DATE"]).sort_index()
    return full_data.drop(columns=["HOSP"])


# Now we can process the Google trends data
relevant_pytrends('topics.txt', step=1, start_year=2020, start_mon=2, stop_year=2020, stop_mon=9, verbose=False)
terms = get_best_topics()
all_google_data = {idx: pd.concat([load_term(key, val, dir="../data/trends/model/", geo=idx, start_year=2020,
                                             start_mon=2, stop_year=2020, stop_mon=9) for key, val in terms.items()],
                                             axis=1) for idx in google_geocodes}
for loc in all_google_data:
    all_google_data[loc]["LOC"] = google_geocodes[loc]
    all_google_data[loc] = all_google_data[loc].reset_index().rename(columns={"date": "DATE"})

all_google_data = pd.concat(all_google_data.values())
all_google_data = all_google_data.groupby(["LOC", "DATE"]).mean()
full_data = all_google_data.join(full_data)

full_data_no_rolling = full_data.copy().dropna()

# Rolling average
full_data = full_data.reset_index()
orig_date = full_data.DATE
full_data = full_data.groupby(['LOC']).rolling(7, center=True).mean().reset_index(0)
full_data['DATE'] = orig_date
full_data = full_data.set_index(["LOC", "DATE"])
full_data = full_data.dropna()

full_data = normalize_hosp_stand(full_data)
full_data_no_rolling = normalize_hosp_stand(full_data_no_rolling)

full_data


# Now we create keras datasets for each LOC
look_back = 1
delay = 7
use_full_valid_test = True
train_ratio = 0.50
valid_ratio = 0.25
test_ratio = 1.0 - train_ratio - valid_ratio


# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - delay):
        a = dataset[i:(i + look_back), :-1]
        # print(a)
        dataX.append(a)
        dataY.append(dataset[i + look_back + delay - 1, -1])
    return np.array(dataX), np.array(dataY)


n_features = -1
full_datapoints = {}
train_datapoints = {}
valid_datapoints = {}
test_datapoints = {}

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

        train_datapoints[loc] = (x[0:train_len], y[0:train_len])
        valid_datapoints[loc] = (x[train_len:train_len + valid_len], y[train_len:train_len + valid_len])
        test_datapoints[loc] = (x[train_len + valid_len:], y[train_len + valid_len:])
else:
    all_locs = list(full_datapoints.keys())
    np.random.shuffle(all_locs)

    length = len(all_locs)
    train_len = int(length * train_ratio)
    valid_len = int(length * valid_ratio)

    train_datapoints = {loc: full_datapoints[loc] for loc in all_locs[0:train_len]}
    valid_datapoints = {loc: full_datapoints[loc] for loc in all_locs[train_len:train_len + valid_len]}
    test_datapoints = {loc: full_datapoints[loc] for loc in all_locs[train_len + valid_len:]}


# # Toy model
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(None, n_features)))
model.add(LSTM(1, return_sequences=True))
model.add(TimeDistributed(Dense(1)))


def train_generator():
    while True:
        for loc in train_datapoints:
            # yield train_datapoints[loc][0].reshape(1, -1, n_features), train_datapoints[loc][1]
            # let's remove some "0" points, just to ensure the LSTM does not directly remember
            # where the peaks are
            d = np.random.randint(0, 20)
            yield train_datapoints[loc][0][d:], train_datapoints[loc][1][d:]


def validation_generator():
    while True:
        for loc in valid_datapoints:
            yield valid_datapoints[loc]


model.compile(loss="mse", optimizer='adam')
history = model.fit(train_generator(), steps_per_epoch=len(train_datapoints), epochs=400, verbose=1, shuffle=False,
                    validation_data=validation_generator(),
                    validation_steps=len(valid_datapoints))


for loc in train_datapoints:
    print("TRAINING", loc)
    x, y = train_datapoints[loc]
    yp = model.predict(x)

    plt.plot(y)
    plt.plot(yp.reshape(-1))
    plt.show()

for loc in valid_datapoints:
    print("VALIDATION", loc)
    x, y = valid_datapoints[loc]
    yp = model.predict(x)

    plt.plot(y)
    plt.plot(yp.reshape(-1))
    plt.show()

for loc in test_datapoints:
    print("TEST", loc)
    x, y = test_datapoints[loc]
    yp = model.predict(x)

    plt.plot(y)
    plt.plot(yp.reshape(-1))
    plt.show()

# # Let's use the validation set


def saved_model(_, _2, _3, _4, p):
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

    def train_generator():
        while True:
            for loc in train_datapoints:
                # yield train_datapoints[loc][0].reshape(1, -1, n_features), train_datapoints[loc][1]
                # let's remove some "0" points, just to ensure the LSTM does not directly remember
                # where the peaks are
                d = np.random.randint(0, 20)
                yield train_datapoints[loc][0][d:], train_datapoints[loc][1][d:]

    def validation_generator():
        while True:
            for loc in valid_datapoints:
                yield valid_datapoints[loc]

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)

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
    history = model.fit(train_generator(), steps_per_epoch=len(train_datapoints), epochs=p["epochs"], verbose=0,
                        shuffle=False,
                        validation_data=validation_generator(),
                        validation_steps=len(valid_datapoints))

    with open('../data/trends/training.log', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    tf.keras.models.save_model(model=model, filepath=save_model)

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

    model = load_model(save_model)

    history.history = pickle.load(open('../data/trends/training.log', "rb"))

    return history, model


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

save_model = "../data/trends/saved_model"

# If no model is saved, we need to train the entire model
if not os.path.exists(save_model):
    scan_object = talos.Scan(
            x=[],
            y=[],
            x_val=[],
            y_val=[],
            params=p,
            model=saved_model,
            experiment_name='trends1',
            fraction_limit=0.01
        )

# If a model is already saved, we load it
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

best_model = scan_object.best_model('val_mse', asc=True)

for name, datapoints in [("Train", train_datapoints), ("Val.", valid_datapoints), ("Test", test_datapoints)]:
    print(f"\\midrule {name}")
    total_mse = 0.0
    total_mae = 0.0
    for loc in datapoints:
        pred = best_model.predict(datapoints[loc][0])
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
for name, datapoints in [("Val.", valid_datapoints), ("Test", test_datapoints)]:
    for loc in datapoints:

        print(name, loc)
        pred = best_model.predict(datapoints[loc][0])
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
