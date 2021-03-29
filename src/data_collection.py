import subprocess
from datetime import date, timedelta
from functools import partial
from subprocess import call
from time import sleep
from calendar import monthrange
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
import pandas as pd
from pytrends.dailydata import get_daily_data
import os.path
from os import listdir
from datetime import date, datetime, timedelta
from my_fake_useragent import UserAgent
import random
import io
import requests
import re
from requests.exceptions import ReadTimeout
import numpy as np
import operator
import traceback
import sys
from typing import List, Iterator, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import util
from copy import deepcopy


# from src.prediction_model import *

# maximum number of days that can be queried before getting a request sampled by week, including the first and last day
# (2020-01-01 -> 2020-01-02 cover 2 days)
max_query_days = 270

google_geocodes = {
    'BE': "Belgique"
}

ua = UserAgent()


def date_parser_daily(x):
    return datetime.strptime(x, "%Y-%m-%d")


def date_parser_hourly(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")


def latest_trends_daily_date():
    """
    :return latest google trends date that can be queried using the daily requests
    """
    latest = date.today() - timedelta(days=3)
    return datetime(latest.year, latest.month, latest.day)


def extract_topics(filename="topics.txt", to_list=False):
    """
    Extracts the pairs of "topics_mid topic_title" in the file provided
    :param filename: file were the topics are written. Can be both a filename or a list of files
    Each valid line must be in the format "topic_mid topic_title"
    :param to_list: whether to return the list of keys or the whole dictionary
    :return: dictionary of {topic_title: topic_mid} for each topic provided
    """
    results = {}
    pattern = "(.+)\s(/[mg]/.+)"
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
    return results if not to_list else list(results.keys())


def compute_corr(left, delta, full_data_be, init_df):
    """
    Computes the correlation between a particular feature and another
    :param left: feature that we want to correlate
    :param delta: delay introduced in the dataframe (in days)
    :param full_data_be: dataframe containing all the data
    :param init_df: initial dataframe
    :return: the correlation between the left feature and the number of new hospitalization
    """
    thisdata = pd.concat([init_df[left].shift(delta), full_data_be["HOSP"]], axis=1)
    return thisdata.corr()["HOSP"][left]


def relevant_pytrends(init_file, start_year: int, start_mon: int, stop_year: int, stop_mon: int,
                      geo: str = 'BE', verbose: bool = True, wait_time: float = 5.0, step=0, threshold=0.8):
    """
    Gets the relevant google trends related to a particular localisation
    :param init_file: path to the file that contains the initial topics to use
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :param geo: geo localisation of the google trends
    :param step: number of iteration for searching related topics. 0 -> return df for the init_file topics/queries only
    :param threshold: minimum threshold (correlation) for a topic to be interesting
    :return: Writes the results in a CSV file
    """
    start_date = date(start_year, start_mon, 1)
    stop_date = get_last_date_of_month(stop_year, stop_mon)
    # initial dict of topics
    init_topics = extract_topics()

    # Interest over time for each topics
    init_df = pd.concat([load_term(key, val) for key, val in init_topics.items()], axis=1)

    # Rolling average over one week --> Smoothing
    init_df = init_df.rolling(7, center=True).mean().dropna()
    data_be = pd.read_csv("be-covid-hospi.csv").groupby(["DATE"]).agg({"NEW_IN": "sum"}).rename(
        columns={"NEW_IN": "HOSP"})  # data for belgium only
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

    out = []  # list containing new most related topics
    for key, val in init_topics.items():
        correlations = [(delay, compute_corr(key, delay, data_be, init_df)) for delay in range(0, 17)]
        best_delay, best_corr = max(correlations, key=lambda x: abs(x[1]))
        out.append((key, val, best_delay, best_corr))
    out = list(filter(lambda x: abs(x[3]) > threshold, out))  # keep only the most correlated topics
    out = [x[0] for x in out]  # keep only the term, not the tuples

    init_df = init_df[out]  # filter the df
    df = pd.DataFrame(columns=("Topic_title", "Topic_mid", "Best delay", "Best correlation"))

    for i in range(step):
        print(f"STEP {i + 1}")
        total_topics = {}
        for term in out:
            pytrends = TrendReq(hl='fr-BE')
            # Initialize build_payload with the word we need data for
            build_payload = partial(pytrends.build_payload,
                                    kw_list=[term], cat=0, geo=geo, gprop='')
            # new topics to evaluate
            new_topics = _fetch_related(pytrends, build_payload, dates_to_timeframe(start_date, stop_date),
                                        term)
            total_topics = {**new_topics, **total_topics}

        # evaluate new topics over time
        list_topics = [load_term(key, val) for key, val in total_topics.items()]
        total_keys = list(total_topics.keys())
        if list_topics:
            related_df = pd.concat(list_topics, axis=1)  # interest over time for each topics
            related_df = related_df.rolling(7, center=True).mean().dropna()  # rolling average
            for key in total_keys:
                if key not in related_df.keys():
                    del total_topics[key]

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

    df.drop_duplicates(subset='Topic_title', keep='first', inplace=True)  # drop duplicates in the dataframe
    df.reset_index(0, inplace=True, drop=True)  # reset the index --> index = date
    df.to_csv('../data/trends/explore/related_topics.csv')  # write results in a csv file


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


def find_correlation_explore(file_hospi="be-covid-hospi.csv"):
    """
    Finds the delay leading to the maximum correlation between every topic in the folder explore and the number of
    new hospitalisations
    :return: Writes the results into a CSV file
    """
    data_be = pd.read_csv(file_hospi).groupby(["DATE"]).agg({"NEW_IN": "sum"}).rename(
        columns={"NEW_IN": "HOSP"})  # data for belgium only
    # Rolling average over one week --> Smoothing + drop the NA values
    data_be = data_be.rolling(7, center=True).mean().dropna()
    # Creation of a dataframe containing the title of the topic, his mid, the best delay and the best correlation
    df = pd.DataFrame(columns=("Topic_title", "Topic_mid", "Best delay", "Best correlation"))

    count = 0
    dir = "../data/trends/explore"
    for file in listdir("../data/trends/explore"):
        search_obj = re.search('BE-WAL-(.*).csv', file)
        if search_obj is not None:
            topic_df = pd.read_csv(os.path.join(dir, file))
            topic_df = topic_df.set_index("date")

            # Add "fake" data (zeroes before the beginning of the crisis) for the hospitalisations
            if count == 0:
                toadd = []
                min_index = data_be.index.min()
                end = datetime.strptime(min_index, "%Y-%m-%d").date()
                min_index_init_df = topic_df.index.min()
                cur = datetime.strptime(min_index_init_df, "%Y-%m-%d").date()
                while cur != end:
                    toadd.append([cur.strftime("%Y-%m-%d"), 0])
                    cur += timedelta(days=1)
                data_be = data_be.reset_index().append(pd.DataFrame(toadd, columns=["DATE", "HOSP"])).set_index("DATE")

            topic_df = topic_df.rolling(7, center=True).mean().dropna()
            correlations = [(delay, compute_corr(topic_df.keys()[0], delay, data_be, topic_df)) for delay in
                            range(0, 17)]
            best_delay, best_corr = max(correlations, key=lambda x: abs(x[1]))
            df.loc[count] = [search_obj.group(1), topic_df.keys()[0], best_delay, best_corr]
            count += 1

    # Write a summary of all the related topics in a CSV file
    df.to_csv('../data/trends/explore/related_topics.csv', index=False)


def get_best_topics(filename='../data/trends/explore/related_topics.csv', number=15):
    """
    Gets the first best topics sorted with respect to the correlation.
    :param filename: path to the file where all the topics generated are stocked
    :param number: number of best topics that we keep
    :return: a dictionary containing topic_title:topic_mid pairs
    """
    df = pd.read_csv(filename)
    # keep the absolute value of the correlation [-1;1] -> [0;1]
    df['Best correlation'] = df['Best correlation'].abs()
    # sort new related topics with respect to correlation
    df.sort_values('Best correlation', ascending=False, inplace=True)
    df = df[['Topic_title', 'Topic_mid']]
    list_topics = df.to_dict('split')['data']

    # keep only number best topics (with the higher correlation)
    dict = {}
    for i in range(min(number, len(list_topics))):
        dict[list_topics[i][0]] = list_topics[i][1]
    return dict


def get_last_date_of_month(year: int, month: int) -> date:
    """
    Given a year and a month returns an instance of the date class
    containing the last day of the corresponding month.
    Source: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
    """
    return date(year, month, monthrange(year, month)[1])


def dates_to_timeframe(start: date, stop: date) -> str:
    """
    Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    :param start: start date
    :param stop: stop date
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """
    Attempts to fecth data and retries in case of a ResponseError.
    :param pytrends: object used for starting the TrendRequest on a localisation
    :param build_payload: object used for initializing a payload containing a particular word
    :param timeframe: string representing the timeframe
    :return a dataframe containing an interest over time for a particular topic
    """
    attempts = 0
    while True:
        try:
            build_payload(timeframe=timeframe)
            return pytrends.interest_over_time()
        except (ResponseError, ReadTimeout) as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break


def _fetch_related(pytrends, build_payload, timeframe: str, term) -> dict:
    """
    Attempts to get the related topics of a particular term and retries in case of ResponseError.
    :param pytrends: object used for starting the TrendRequest on a localisation
    :param build_payload: object used for initializing a payload containing a particular word
    :param timeframe: string representing the timeframe
    :return : a dictionary of the related topics
    """
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

    term = term.replace('(', ' ').replace(')', '').replace("'", '')

    if df[term]['rising'].empty:
        return {}

    dic_rising = df[term]['rising'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()['topic_mid']
    dic_top = df[term]['top'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()['topic_mid']
    return {**dic_rising, **dic_top}


def merge_trends_batches(left, right, overlap_time, topic, is_hour=True, verbose=False, drop=None):
    """
    return the concatenation of left and right, correctly scaled based on their overlap (in hours or days)

    :param left: accumulator dataframe
    :param right: new dataframe to add to the accumulator
    :param overlap_time: number of time points that are overlapping
    :param topic: topic code considered
    :param is_hour: if true, the overlap is computed in hours. Else in days
    :param verbose: True if information must be printed
    :param drop: can be one of None, 'left' or 'right'. The function can choose to keep scaled values
        from left or right on the intersection. If drop == left -> keep the right values, if drop == right ->
        keep the left values. Else take whatever value.
    """
    if left.empty:
        return right
    # retrieve the overlapping points:
    overlap_start = right.index.min()
    if is_hour:
        overlap_end = overlap_start + timedelta(hours=overlap_time - 1)
    else:
        overlap_end = overlap_start + timedelta(days=overlap_time - 1)
    left_overlap = left[overlap_start:overlap_end]
    right_overlap = right[overlap_start:overlap_end]
    # drop 0 and inf values on the overlap
    scaling_full = (left_overlap[topic] / right_overlap[topic]).replace([0, np.inf, -np.inf], np.nan).dropna()
    if verbose:
        print('overlap left:', left_overlap[topic])
        print('overlap right:', right_overlap[topic])
        print('ratio:', scaling_full)
        print("my drop=", drop)
    scaling = scaling_full.mean()
    if scaling < 1:  # right has not the good scale
        if drop == 'left':
            left_to_keep = left[left.index < overlap_start]
            right_to_add = right * scaling
            scaled = left_to_keep.append(right_to_add)
        else:
            right_to_add = right[right.index > overlap_end]
            right_to_add = right_to_add * scaling
            scaled = left.append(right_to_add)
    else:  # left has not the good scale
        if drop == 'right':
            right_to_keep = right[right.index > overlap_end]
            left_to_add = left / scaling
            scaled = left_to_add.append(right_to_keep)
        else:
            left_to_add = left[left.index < overlap_start]
            left_to_add = left_to_add / scaling
            scaled = left_to_add.append(right)
    if np.any(scaled.index.duplicated()):
        print("ho")
    return scaled


def collect_historical_interest(topic_mid: str, topic_title: str, geo: str, begin_tot: datetime = None,
                                end_tot: datetime = None, overlap_hour: int = 15, verbose: bool = True) -> pd.DataFrame:
    """
    load collect and save hourly trends data for a given topic over a certain region

    :param topic_mid: mid code
    :param topic_title: title of the topic
    :param geo: google geocode
    :param begin_tot: beginning date. If None, default to 01/02/2020
    :param end_tot: end date. If None, default to today
    :param overlap_hour: number of overlapping point
    :param verbose: whether to print information while the code is running or not

    :return dataframe of collected data
    """
    dir = "../data/trends/collect/"
    batch_column = 'batch_id'
    hour_format = "%Y-%m-%dT%H"
    file = f"{dir}{geo}-{topic_title}.csv"
    min_delta = timedelta(days=3)
    if end_tot is None:  # if not specified, get the latest day with 24 hours for data
        end_tot = datetime.now().replace(microsecond=0, second=0, minute=0)
        if end_tot.hour != 23:
            end_tot = end_tot.replace(hour=23) - timedelta(days=1)

    if os.path.exists(file):  # load previous file
        date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        df_tot = pd.read_csv(file, parse_dates=['date'], date_parser=date_parser).set_index('date')
        idx_max = df_tot.index.max()
        if end_tot == df_tot.index.max():
            return df_tot
        max_batch = df_tot[batch_column].max()
        if len(df_tot.loc[df_tot[batch_column] == max_batch]) < 192:  # if the last batch was not done on a whole week
            df_tot = df_tot.loc[df_tot[batch_column] < max_batch]  # drop the last batch
        i = df_tot[batch_column].max() + 1  # id of the next batch
        begin_tot = df_tot.index.max() - timedelta(hours=(overlap_hour-1))
    else:
        df_tot = pd.DataFrame()
        if begin_tot is None:
            begin_tot = datetime.strptime("2020-02-01T00", hour_format)
        i = 0

    begin_cur = begin_tot  # beginning of current batch
    end_cur = begin_tot + timedelta(days=7, hours=23)  # end of current batch
    if end_cur > end_tot:  # end of date
        end_cur = end_tot
        if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
            begin_cur = end_cur - min_delta

    delta = timedelta(days=7, hours=23) - timedelta(hours=(overlap_hour-1))  # diff between 2 batches
    delay = 0
    trials = 0
    finished = False
    if verbose:
        print(f"topic {topic_title} geo {geo}")
    while not finished:
        timeframe = begin_cur.strftime(hour_format) + " " + end_cur.strftime(hour_format)
        try:
            sleep(delay + random.random())
            if verbose:
                print(f"downloading {timeframe} ... ", end="")
            agent = ua.random()
            pytrends = TrendReq(hl="fr-BE")
            pytrends.build_payload([topic_mid], geo=geo, timeframe=timeframe, cat=0)
            df = pytrends.interest_over_time()
            if df.empty:
                df = pd.DataFrame(data={'date': pd.date_range(start=begin_cur, end=end_cur, freq='H'), topic_mid: 0}).set_index('date')
                df[batch_column] = -i
            else:
                df.drop(columns=['isPartial'], inplace=True)
                df[batch_column] = i
            i += 1
            df_tot = df_tot.append(df)
            df_tot.to_csv(file)
            if end_cur == end_tot:
                finished = True
            begin_cur += delta

            if end_cur + delta > end_tot:  # end of date
                end_cur = end_tot
                if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                    begin_cur = end_cur - min_delta
            else:  # not end of date, increment
                end_cur = end_cur + delta

            if verbose:
                print("loaded")
            trials = 0
        except ResponseError as err:  # use a delay if an error has been received
            if str(err.response) == '<Response [500]>':
                write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                f = open(write_file, "a+")
                f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                print(f"Error 500. Timeframe not available")
                if end_cur == end_tot:
                    finished = True
                begin_cur += delta

                if end_cur + delta > end_tot:  # end of date
                    end_cur = end_tot
                    if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                        begin_cur = end_cur - min_delta
                else:  # not end of date, increment
                    end_cur = end_cur + delta
            else:
                trials += 1
                delay = 60
                if trials > 3:
                    write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                    f = open(write_file, "a+")
                    f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                    print("ReadTimeOut. Timeframe not available")
                    trials = 0
                    if end_cur == end_tot:
                        finished = True
                    begin_cur += delta

                    if end_cur + delta > end_tot:  # end of date
                        end_cur = end_tot
                        if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                            begin_cur = end_cur - min_delta
                    else:  # not end of date, increment
                        end_cur = end_cur + delta
                if verbose:
                    print(f"Error when downloading (ResponseError). Retrying after sleeping during {delay} sec ... Trial : {trials}")
        except ReadTimeout:
            trials += 1
            delay = 60
            if trials > 3:
                write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                f = open(write_file, "a+")
                f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                print("ReadTimeOut. Timeframe not available")
                trials = 0
                if end_cur == end_tot:
                    finished = True
                begin_cur += delta

                if end_cur + delta > end_tot:  # end of date
                    end_cur = end_tot
                    if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                        begin_cur = end_cur - min_delta
                else:  # not end of date, increment
                    end_cur = end_cur + delta
            if verbose:
                print(f"Error when downloading (ReadTimeout). Retrying after sleeping during {delay} sec ... Trial : {trials}")
    return df_tot


def get_historical_interest_normalized(topic, date_begin, date_end, geo, overlap_hour=1, verbose=True, sleep_fun=None):
    """
    return the historical data for a topic, hour by hour, on the given interval.

    the normalization is performed based on overlap_hour common data points, in order ot have only one value of 100 on
    the final dataframe

    :param topic: topic to query
    :param date_begin: first date to search
    :param date_end: last date to search
    :param geo: geocode of where we want to retrieve the data
    :param overlap_hour: hours used for overlap between 2 requests
    :param verbose: whether to print information while the code is running or not
    :param sleep_fun: 0-argument function, telling how many seconds we must wait between each request
    :return: normalized dataframe for the topic
    """
    start = datetime.strptime(date_begin, "%Y-%m-%d")
    end = datetime.strptime(date_end, "%Y-%m-%d")
    if sleep_fun is None:  # no sleep function specified
        sleep_fun = lambda: 10 + 5 * random.random()  # by default sleep for 10 seconds + between 0 and 5 randomly
    delta = timedelta(days=7, hours=23) - timedelta(hours=(overlap_hour - 1))  # delay between each batch
    start_batch = start - delta
    end_batch = start - delta + timedelta(days=7, hours=23)
    finished = False
    i = 0
    interest_unscaled = pd.DataFrame()
    interest_scaled = pd.DataFrame()
    while not finished:
        start_batch = start_batch + delta
        end_batch = min(end_batch + delta, end)
        timeframe = start_batch.strftime("%Y-%m-%dT%H") + ' ' + end_batch.strftime("%Y-%m-%dT%H")
        pytrends = TrendReq(hl='fr-BE')
        build_payload = partial(pytrends.build_payload, kw_list=[topic], cat=0, geo=geo, gprop='')
        batch = _fetch_data(pytrends, build_payload, timeframe)
        if verbose:  # print some information
            if batch.empty or 0 in batch[topic].values:
                print(f"dataframe contains zero or is empty :(")
            else:
                print("dataframe does not contain any zero")
            print(f"iter {i} timeframe {timeframe}")
        batch.drop(columns=['isPartial'], inplace=True)
        interest_unscaled = interest_unscaled.append(batch)  # add the new data
        # we have overlap_hour points that we can use to normalize the data
        interest_scaled = merge_trends_batches(interest_scaled, batch, overlap_hour, topic)
        if end_batch == end:  # last date was visited, end of loop
            finished = True
        else:
            sleep(sleep_fun())  # prevent 429s
        i += 1
    return interest_unscaled.reset_index(drop=True), interest_scaled


def get_daily_data(word: str, start_year: int, start_mon: int, stop_year: int, stop_mon: int,
                   geo: str = 'BE', verbose: bool = True, wait_time: float = 5.0) -> pd.DataFrame:
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

    # TODO : Faire en sorte de récolter les données en jours lorsque l'API nous les donne en semaine
    if (stop_date - start_date).days >= 365:
        # Obtain monthly data for all months in years [start_year, stop_year]
        monthly = _fetch_data(pytrends, build_payload,
                              dates_to_timeframe(start_date, stop_date))

        # Get daily data, month by month
        results = {}
        # if a timeout or too many requests error occur we need to adjust wait time
        current = start_date
        while current < stop_date:
            last_date_of_month = get_last_date_of_month(current.year, current.month)
            timeframe = dates_to_timeframe(current, last_date_of_month)
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
                              dates_to_timeframe(start_date, stop_date))

    """ dataframe contains
    - word_unscaled: data with a top 0-100 monthly
    - word_monthly: data with a top 0-100 for start_date to end_date
    - scale: word_monthly/100
    - word: word_unscaled * scale -> contains data between 0-100 (with only one 100)
    """
    if verbose:
        print(f'{word} downloaded from google trends')
    return complete


def _dl_term(term, geo="BE-WAL", start_year=2020, start_mon=2, stop_year=2020, stop_mon=9):
    """
    Gets daily data for a timeframe and a particular term. By default, the trends requests are done for Belgium.
    :param term: topic we want to fetch daily data for.
    :param geo: geo localisation for which the request is done
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :return: a dataframe containing the results of the trends requests for the term
    """
    df = get_daily_data(term, start_year=start_year, start_mon=start_mon, stop_year=stop_year, stop_mon=stop_mon,
                        geo=geo, verbose=False)
    if df.empty:
        return df
    else:
        return df[term].copy()


def load_term(termname, term, dir="../data/trends/explore/", geo="BE-WAL", start_year=2020, start_mon=2,
              stop_year=2020, stop_mon=9):
    """
    Loads the results of the trends request in a CSV file for each topic.
    :param termname: name of the topic we want to evaluate with Google trends
    :param term: mid of the topic we want to evaluate with Google trends
    :param dir: directory where we will load the CSV files
    :param geo: geo localisation of the trends request
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :return: a dataframe containing the evaluation of the trends request for a particular term
    """
    if "/" in termname:
        termname_path = termname.replace("/", "-")
    else:
        termname_path = termname

    path = f"{dir}{geo}-{termname_path}.csv"
    encoded_path = requests.get(path).content

    if not os.path.exists(path):
        print(f"DL {geo} {termname}")
        content = _dl_term(term, start_year=start_year, start_mon=start_mon, stop_year=stop_year, stop_mon=stop_mon)
        if content.empty:
            return content
        content.to_csv(io.StringIO(encoded_path.decode("utf-8")))

    content = pd.read_csv(io.StringIO(encoded_path.decode("utf-8")))
    content = content.rename(columns={term: termname})
    content = content.set_index("date")
    return content


def create_dataframe_belgium(hospi_belgium='../data/hospi/be-covid-hospi.csv'):
    data_be = pd.read_csv(hospi_belgium).groupby(["DATE"]).agg({"NEW_IN": "sum"}).reset_index().rename(
        columns={"NEW_IN": "HOSP"})

    data_be["LOC"] = "Belgique"
    base_date = datetime.strptime("2020-02-01", "%Y-%m-%d").date()
    end = datetime.strptime(data_be['DATE'].min(), "%Y-%m-%d").date()
    cur = base_date
    toadd = []
    while cur != end:
        toadd.append([cur.strftime("%Y-%m-%d"), 'Belgique', 0])
        cur += timedelta(days=1)
    data_be = data_be.append(pd.DataFrame(toadd, columns=["DATE", "LOC", "HOSP"])).set_index(["LOC", "DATE"])
    data_be = data_be.sort_index()
    list_hospi = data_be['HOSP']
    new_list_hospi = [0]
    for i in range(1, len(list_hospi)):
        diff = list_hospi[i] - list_hospi[i-1]
        if diff == 0 or list_hospi[i] == 0:
            new_list_hospi.append(0)
        else:
            if list_hospi[i-1] == 0 and list_hospi[i] != 0:
                new_list_hospi.append(1.0)
            else:
                diff = diff / list_hospi[i-1]
                new_list_hospi.append(diff)
    # Normalization between 0 and 1
    new_list_hospi = [(x-min(new_list_hospi))/(max(new_list_hospi)-min(new_list_hospi)) for x in new_list_hospi]
    data_be['HOSP'] = new_list_hospi
    print(data_be)
    return data_be


def create_dataframe(hospi_france='hospitals.csv', hospi_belgium='be-covid-hospi.csv',
                     department_france='france_departements.csv'):
    """
    Creates the dataframe containing the number of daily new hospitalizations
    with respect to the date and the localisation (FR and BE)
    """
    departements = pd.read_csv(department_france)
    hospitalisations = pd.read_csv(hospi_france, sep=";")
    data_fr = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(["regionName", "jour"],
                                                                                                as_index=False).agg(
        {"incid_hosp": "sum"})
    data_fr = data_fr.rename(columns={"jour": "DATE", "incid_hosp": "HOSP", "regionName": "LOC"})

    # Belgian data
    data_be = pd.read_csv(hospi_belgium).groupby(["DATE"]).agg({"NEW_IN": "sum"}).reset_index().rename(
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
    return full_data


def normalize_hosp_stand(full_data):
    """
    Normalizes and standardizes the number of new hospitalizations between 0 and 1 per localisation.
    The goal is to predict peaks/modification of the slope, not numbers.
    :param full_data: dataframe containing the trends for each symptom and the number of new hospitalizations
    with respect to the date and the localisation
    :return the same dataframe than full_data without the row of the hospitalizations
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


def google_trends_process(full_data, terms, start_year, start_mon, stop_year, stop_mon, step=1, data_collection=True):
    """
    Processes the Google trends data : drop NA values, smoothing, set/reset index and normalization
    :param full_data: dataframe containing the trends for each symptom and the number of new hospitalizations
    with respect to the date and the localisation
    :param terms: dictionary of topics to use
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :param step: number of iteration for finding new related topics
    """
    find_correlation_explore()

    # If we need to collect new data
    if data_collection:
        relevant_pytrends('topics.txt', step=step, start_year=start_year, start_mon=start_mon, stop_year=stop_year,
                          stop_mon=stop_mon, verbose=False)

    if terms is None:
        terms = get_best_topics()

    url_trends = "https://raw.githubusercontent.com/gerardmargaux/CovidThesis/master/data/trends/model/"
    all_google_data = {idx: pd.concat([load_term(key, val, dir=url_trends, geo=idx, start_year=start_year,
                                                 start_mon=start_mon, stop_year=stop_year, stop_mon=stop_mon) for
                                       key, val in terms.items()], axis=1) for idx in google_geocodes}

    for loc in all_google_data:
        all_google_data[loc]["LOC"] = google_geocodes[loc]
        all_google_data[loc] = all_google_data[loc].reset_index().rename(columns={"index": "DATE", 'date': 'DATE'})

    all_google_data = pd.concat(all_google_data.values())
    all_google_data = all_google_data.groupby(["LOC", "DATE"]).mean()
    full_data = all_google_data.join(full_data)

    full_data_no_rolling = full_data.copy().dropna()

    full_data = full_data.reset_index()
    orig_date = full_data.DATE
    # Rolling average on one week --> smoothing
    full_data = full_data.groupby(['LOC']).rolling(7, center=True).mean().reset_index(0)
    full_data['DATE'] = orig_date
    full_data = full_data.set_index(["LOC", "DATE"])
    full_data = full_data.dropna()  # drop NA values

    # Normalization and standardization of the number of new hospitalizations
    full_data = normalize_hosp_stand(full_data)
    full_data_no_rolling = normalize_hosp_stand(full_data_no_rolling)

    return full_data, full_data_no_rolling


'''
def actualize_trends(keywords: dict, verbose=True, start_year=2020, start_month=3, path='../data/trends/model'):
    """
    get the latest available data from google trends and stores it as csv files
    :param keywords: dict of topic_title:topic_mid
    :param verbose: True if information must be printed over time
    :param start_year: year of the start date
    :param start_month: month of the start date
    :param path: path where the trends are stored
    """
    geocodes = {
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

    today = date.today()  # take the latest data
    stop_year = today.year
    stop_month = today.month
    first_iteration = True
    asked_min = date(start_year, start_month, 1)
    asked_max = today
    for geo, description in geocodes.items():
        if verbose:
            print(f'-- collecting data for {geo}:{description} --')
        for name, code in keywords.items():
            # get the data for the corresponding topic in the corresponding localisation
            csv_file = f'{path}/{geo}-{name}.csv'
            if os.path.exists(csv_file):  # check if an existing file already contain the dates
                df = pd.read_csv(csv_file)
                df.set_index('date', inplace=True)
                stored_max = datetime.strptime(df.index.max().replace(',', ''), '%Y-%m-%d').date()
                stored_min = datetime.strptime(df.index.min().replace(',', ''), '%Y-%m-%d').date()
                if stored_max >= asked_max and stored_min <= asked_min:
                    continue  # no data is downloaded if the range asked is already there
            df = get_daily_data(code, start_year, start_month, stop_year, stop_month, geo, verbose=verbose)
            # sleep(1 + random.random())  # prevent 429s
            if first_iteration:  # google trends might not have the data for today
                # store the latest day where trends data exist
                asked_max = datetime.strptime(str(df.index.max()).replace(',', ''), '%Y-%m-%d 00:00:00').date()
                first_iteration = False
            df.drop(columns=['isPartial'], inplace=True)
            df.to_csv(csv_file)
'''

def actualize_hospi():
    url_hospi_belgium = "https://raw.githubusercontent.com/pschaus/covidbe-opendata/master/static/csv/be-covid-hospi.csv"
    url_hospi_france_new = "https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c"
    url_hospi_france_tot = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"
    url_hospi_world = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    # hospi in belgium
    encoded_path_be = requests.get(url_hospi_belgium).content
    df_hospi_be = pd.read_csv(io.StringIO(encoded_path_be.decode("utf-8"))).drop(axis=1, columns='Unnamed: 0')
    df_hospi_be.to_csv('../data/hospi/be-covid-hospi.csv', index=False)

    # hospi in France
    encoded_path_fr = requests.get(url_hospi_france_tot).content
    df_hospi_fr = pd.read_csv(io.StringIO(encoded_path_fr.decode("utf-8")))
    df_hospi_fr = df_hospi_fr.rename(columns=lambda s: s.replace('"', ''))
    for i, col in enumerate(df_hospi_fr.columns):
        df_hospi_fr.iloc[:, i] = df_hospi_fr.iloc[:, i].str.replace('"', '')
    df_hospi_fr.to_csv('../data/hospi/fr-covid-hospi-total.csv', index=False)

    # new hospi in France
    encoded_path_fr_new = requests.get(url_hospi_france_new).content
    df_hospi_fr_new = pd.read_csv(io.StringIO(encoded_path_fr_new.decode("utf-8")))
    df_hospi_fr_new = df_hospi_fr_new.rename(columns=lambda s: s.replace('"', ''))
    for i, col in enumerate(df_hospi_fr_new.columns):
        df_hospi_fr_new.iloc[:, i] = df_hospi_fr_new.iloc[:, i].str.replace('"', '')
    df_hospi_fr_new.to_csv('../data/hospi/fr-covid-hospi.csv', index=False)

    # hospi in the world
    encoded_path_world = requests.get(url_hospi_world).content
    df_world = pd.read_csv(io.StringIO(encoded_path_world.decode("utf-8")))
    df_world = df_world[['iso_code', 'location', 'date', 'icu_patients', 'hosp_patients', 'population']]
    df_world.drop(columns=['population']).to_csv('../data/hospi/world.csv', index=False)

    # also write the world population
    pop = df_world.groupby("iso_code").agg("population").mean().to_dict()
    with open("../data/population.txt", "w") as file:
        file.write(json.dumps(pop))
    return


def actualize_github():
    subprocess.run("git pull", shell=True)
    file_list = [
        '../data/hospi/fr-covid-hospi-total.csv',
        '../data/hospi/be-covid-hospi.csv',
        '../data/trends/model'
    ]
    for file in file_list:
        subprocess.run(f'git add {file}', shell=True)
    commit_message = "'Automatic actualization of trends and hospitalizatons'"
    subprocess.run(f'git commit -m {commit_message}', shell=True)
    subprocess.run(f'git push', shell=True)


def largest_dates_iterator(begin: datetime, end: datetime, number: int) -> Tuple[datetime, datetime]:
    """
    gives the largest interval of dates that will be used by dates_iterator
    :param begin: date of beginning for the query
    :param end: date of end for the query
    :param number: amount of queries that must cover this interval
    :return largest interval of dates that will be used
    """
    min_date = datetime.today()
    max_date = datetime.strptime("1990-01-01", "%Y-%m-%d")
    for date_a, date_b in dates_iterator(begin, end, number):
        min_date = min(min_date, date_a)
        max_date = max(max_date, date_b)
    return min_date, max_date


def dates_iterator(begin: datetime, end: datetime, number: int) -> Iterator[Tuple[datetime, datetime]]:
    """
    return the largest interval of dates that must be available to provide number queries on the interval
    :param begin: date of beginning for the query
    :param end: date of end for the query
    :param number: amount of queries that must cover this interval
    :return yield number tuples of dates (a, b), that cover [begin, end]
    """
    # maximum date allowed for google trends, included
    max_end = latest_trends_daily_date()
    number_given = 0
    lag_left = 0
    lag_right = 0
    if end > max_end:  # impossible to provide the queries
        return []
    while number_given < number:
        i = 0
        while number_given < number and i <= lag_right:
            wanted_end = end + timedelta(days=i)
            if wanted_end <= max_end:
                number_given += 1
                yield begin - timedelta(days=lag_left), wanted_end
            i += 1

        wanted_end = end + timedelta(days=lag_right)
        if wanted_end <= max_end:
            j = lag_left-1
            while number_given < number and j >= 0:
                number_given += 1
                yield begin - timedelta(days=j), wanted_end
                j -= 1

        lag_left += 1
        lag_right += 1


def mean_query(number: int, begin: datetime, end: datetime, topic: str, geo: str, cat=0, verbose=True):
    """
    provide multiple queries on the period begin->end. the column topic contains the mean of the queries
    the queries use different interval in order to provide different results
    """
    df_tot = pd.DataFrame()
    cnt = 0
    pytrends = TrendReq(retries=2, backoff_factor=0.1)

    for k, (begin_tmp, end_tmp) in enumerate(dates_iterator(begin, end, number)):
        timeframe = dates_to_timeframe(begin_tmp, end_tmp)
        # Initialize build_payload with the word we need data for
        build_payload = partial(pytrends.build_payload,
                                kw_list=[topic], cat=cat, geo=geo, gprop='')
        if verbose:
            print(f"timeframe= {timeframe} ({k + 1}/{number})")
        df = _fetch_data(pytrends, build_payload, timeframe)
        df = df[begin:end]
        if 100 not in df[topic]:
            df[topic] = df[topic] * 100 / df[topic].max()
        df_tot[f"{topic}_{cnt}"] = df[topic]
        cnt += 1
        if cnt >= number:
            df_tot[topic] = df_tot.mean(axis=1)
            df_tot[topic] = 100 * df_tot[topic] / df_tot[topic].max()
            return df_tot


def timeframe_normalize_clusters(list_df, overlap=30):
    """
    take as input the list of df given by sanitize_hourly_data and return the list of dates needed to normalize
    the whole set of data, as tuple of dates
    """
    list_tf = []
    delta = timedelta(days=overlap)
    max_end = latest_trends_daily_date()
    for df_i, df_j in zip(list_df, list_df[1:]):
        begin = (df_i.index.max() - delta).to_pydatetime()
        end = (df_j.index.min() + delta).to_pydatetime()
        if end > max_end:
            end = max_end
        list_tf.append((begin, end))
    return list_tf


def drop_incomplete_days(list_df: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    filter a list of hourly dataframes, to return a list where each dataframe:
    - begins at MM-DD-0h
    - ends at MM-DD-23h
    - contains at least 3 days of data
    - has at least 3 points of value > 10
    - has less than 10 consecutive values of 0 at the end / beginning
    :param list_df: list of dataframes to filter
    """
    result = []
    for i in range(len(list_df)):
        df = list_df[i]
        old_begin, old_end = df.index.min(), df.index.max()
        new_begin = old_begin + timedelta(hours=((24 - old_begin.hour) % 24))  # MM-DD-0h
        new_end = old_end - timedelta(hours=((old_end.hour + 1) % 24))  # MM-DD-23h
        cur_df = df[new_begin:new_end]
        # check for chain of zeros at the beginning and the end
        has_zero = True
        hours_drop = 10  # consecutive values to check
        delta = timedelta(hours=hours_drop)
        while has_zero and new_begin < new_end:  # zeros at the beginning
            if cur_df[new_begin:new_begin + delta].sum()[0] == 0:
                new_begin += timedelta(days=1)
            else:
                has_zero = False

        has_zero = True
        while has_zero and new_begin < new_end:  # zeros at the end
            if cur_df[new_end - delta:new_end].sum()[0] == 0:
                new_end -= timedelta(days=1)
            else:
                has_zero = False
        # new dates for the dataframe
        cur_df = cur_df[new_begin:new_end]
        # check if the resulting dataframe can be added
        if not cur_df.empty and (new_end - new_begin).days >= 2 and len(np.where(cur_df > 10)[0]) > 3:
            result.append(cur_df)
    return result


def sanitize_hourly_data(df: pd.DataFrame, topic_code: str) -> List[pd.DataFrame]:
    """
    sanitize hourly data, transforming it to a list of daily data and removing missing values
    :param df: dataframe of hourly data on a trends topic, indexed by hourly dates
    :param topic_code: code for the topic
    :return list of data sanitized: missing values are removed, leading to dataframes with holes between one another
    """
    list_df_hourly = scale_df(df, topic_code)  # scale the dataframe
    list_df_hourly = drop_incomplete_days(list_df_hourly)  # drop the incomplete days (check doc for details)
    list_df_hourly = [df.resample('D').mean() for df in list_df_hourly]  # aggregate to daily data
    return list_df_hourly


def find_largest_intersection(df_a: pd.DataFrame, df_b: pd.DataFrame, list_df_daily: List[pd.DataFrame], overlap: int = 30)\
        -> Tuple[pd.DataFrame, bool]:
    """
    find daily dataframe with the largest intersection on df_a and df_b
    :param df_a: first dataframe
    :param df_b: second dataframe
    :param list_df_daily: list of dataframe to consider in order to find the one with the largest intersection
    :param overlap: number of overlap that should be used for the intersection
    """
    if not list_df_daily:  # no list of daily dataframe given
        return pd.DataFrame(), True

    best_inter = -1
    best_df = None
    can_be_actualized = True  # true if the largest date must be actualized
    max_date = latest_trends_daily_date()
    max_overlap_left = min(len(df_a), overlap)  # upper bound considered for the overlap
    max_overlap_right = min(len(df_b), overlap)
    for df_candidate in list_df_daily:
        intersection_left = len(df_a.index.intersection(df_candidate.index))
        intersection_right = len(df_b.index.intersection(df_candidate.index))
        inter = min(intersection_left, intersection_right)
        if inter >= best_inter:
            best_df = df_candidate
            best_inter = inter
            if intersection_right < max_overlap_right and df_candidate.index.max() < max_date:  # new data is available
                can_be_actualized = True
            elif intersection_left < max_overlap_left:  # better data can be found
                can_be_actualized = True
            else:
                can_be_actualized = False
    return best_df, can_be_actualized


def daily_gap_and_model_data(geo: str, topic_title: str, topic_mid: str, number: int = 20, overlap: int = 30,
                      verbose: bool = True, refresh: bool = True) -> pd.DataFrame:
    """
    collect the daily data on the gap where the hourly data could not be retrieved
    if the daily data is outdated or absent, it is collected and saved to a csv file
    :param geo: geocode that must be queried
    :param topic_title: title of the topic to query
    :param topic_mid: code of the topic to query
    :param number: number of daily queries to do on the gaps
    :param overlap: number of overlapping points between a daily request and the 2 hourly requests around it
    :param verbose: whether to display information while running or not
    :param refresh: whether to find new daily gap or use only the existing ones
    :return model dataframe
    """
    data_daily_dir = "../data/trends/collect_gap"
    data_hourly_dir = "../data/trends/collect"
    data_model_dir = "../data/trends/model"
    # retrieve the hourly data
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_hourly = pd.read_csv(f"{data_hourly_dir}/{geo}-{topic_title}.csv", parse_dates=['date'],
                            date_parser=date_parser).set_index('date')
    # transform the hourly data to a list of daily data
    list_df_hourly = sanitize_hourly_data(df_hourly, topic_mid)
    # collect the existing daily data
    # pattern = "([A-Z]{2}(?:-[A-Z])?)-(\D*)-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.csv"
    starting_pattern = f"{geo}-{topic_title}-"
    existing_files = [filename for filename in os.listdir(data_daily_dir) if filename.startswith(starting_pattern)]
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    list_daily_df = [pd.read_csv(f"{data_daily_dir}/{file}", parse_dates=['date'],
                                 date_parser=date_parser).set_index('date')[[topic_mid]] for file in existing_files]
    if verbose:
        for i, df in enumerate(list_df_hourly):
            print(f"hourly agg data: cluster {i}: {df.index.min().to_pydatetime().date()} -> {df.index.max().to_pydatetime().date()}")
    # check if the gaps can be covered with the existing daily requests saved
    list_dates_actualize = []
    list_daily_intersection = []
    for df_a, df_b in zip(list_df_hourly, list_df_hourly[1:]):
        df_intersection, can_be_actualized = find_largest_intersection(df_a, df_b, list_daily_df, overlap=overlap)
        if can_be_actualized and refresh:  # new data can be queried and must be refreshed
            # overlap-1 because the date is included
            dates_query = (df_a.index.max() - timedelta(days=overlap-1)).to_pydatetime(), \
                          min((df_b.index.min() + timedelta(days=overlap-1)).to_pydatetime(), latest_trends_daily_date())
            list_dates_actualize.append(dates_query)
        else:  # the query respect the expected overlap
            list_daily_intersection.append(df_intersection)
    files_to_remove = []
    for i, df in enumerate(list_daily_df):
        to_remove = True
        for df_chosen in list_daily_intersection:
            if df is df_chosen:
                to_remove = False
                break
        if to_remove:
            files_to_remove.append(existing_files[i])

    # do the request for the dataframes to actualize
    min_dates = find_min_dates_queries(list_dates_actualize, number)
    if min_dates and verbose:
        print("querying new interval")
    for begin, end in min_dates:
        df = collect_holes_data(topic_mid, topic_title, number, begin, end, geo, cat=0, verbose=verbose)
        list_daily_intersection.append(df[[topic_mid]])

    # remove the old daily dataframes that are not used anymore
    for file in files_to_remove:
        if verbose:
            print(f"removing {data_daily_dir}/{file}")
        os.remove(f"{data_daily_dir}/{file}")

    # form the model data and save it to csv
    complete_df = merge_hourly_daily(list_df_hourly, list_daily_intersection, topic_mid, drop=True)
    filename = f"{data_model_dir}/{geo}-{topic_title}.csv"
    complete_df.to_csv(filename)
    return complete_df


def rescale_batch(df_left, df_right, df_daily, topic_code, overlap_max=30, rolling=7, overlap_min=2, drop=True):
    """
    rescale a left and a right batch with a hole in between, covered by df_daily
    :param df_left: DataFrame on the left interval
    :param df_right: DataFrame on the right interval
    :param df_daily: DataFrame with data between df_left and df_right, used for the merge
    :param overlap_max: maximum number of datapoints used for the overlap
    :param rolling: rolling average on the df_daily data
    :param overlap_min: minimum number of points on the intersection allowed to accept a rolling average.
        If the rolling average provide less points, df_daily is used instead of the rolling data.
    :param drop: whether to drop preferabily the df_daily data on the interval or not
    :return batch_rescaled: DataFrame of data between df_left.min and df_right.max
    """
    if drop:
        drop = ['right', 'left']
    else:
        drop = [None, None]

    daily_rolling = df_daily.rolling(rolling, center=True).mean().dropna()
    daily_rolling = 100 * daily_rolling / daily_rolling.max()
    overlap = df_left.index.intersection(daily_rolling.index)
    overlap_right = daily_rolling.index.intersection(df_right.index)
    # print(len(overlap))
    # print(len(overlap_right))
    if len(overlap) < overlap_min or len(overlap_right) < overlap_min:
        overlap = df_left.index.intersection(df_daily.index)
        daily_used = df_daily
    else:
        daily_used = daily_rolling

    if len(overlap) > overlap_max:
        overlap = overlap[-overlap_max:]
    overlap_len = len(overlap)
    title = daily_used.columns[0]
    daily_used.loc[daily_used[title] < 0.01, title] = 0
    # print("overlap left len", overlap_len)
    # print("left")
    # plot_trends(df_left, df_left.columns[0])
    # print("daily")
    # plot_trends(daily_used, df_left.columns[0])
    batch_rescaled = merge_trends_batches(df_left, daily_used[overlap.min():], overlap_len, topic_code,
                                          is_hour=False, verbose=False, drop=drop[0])
    batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
    overlap = batch_rescaled.index.intersection(df_right.index)
    if len(overlap) > overlap_max:
        overlap = overlap[:overlap_max]
    overlap_len = len(overlap)
    # print("overlap right len", overlap_len)
    # print("temporary merge")
    # plot_trends(batch_rescaled, batch_rescaled.columns[0])
    # print("right")
    # plot_trends(df_right, df_left.columns[0])
    batch_rescaled = merge_trends_batches(batch_rescaled[:overlap.max()], df_right, overlap_len, topic_code,
                                          is_hour=False, verbose=False, drop=drop[1])
    batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
    # print("merge final")
    # plot_trends(batch_rescaled, batch_rescaled.columns[0])
    return batch_rescaled


def merge_hourly_daily(list_df_hourly: List[pd.DataFrame], list_df_daily: List[pd.DataFrame], topic_code: str,
                       drop: bool, add_daily_end=True):
    """
    merge the hourly (deterministic) aggregated batches, using the daily (stochastic) batches on the missing interval
    :param list_df_hourly: sorted list of deterministic DataFrame, having a daily index
    :param list_df_daily: list of stochastic DataFrame, having data on the missing interval of list_df_hourly
    :param topic_code: topic code
    :param drop: whether to drop the stochastic data preferably or not
    :param add_daily_end: if True, add daily data data at the end if the max date of daily data > max date of hourly data
    :return df: merged DataFrame
    """

    for i, df_right in enumerate(list_df_hourly):
        if i == 0:
            df = df_right
        else:
            df_daily, _ = find_largest_intersection(df, df_right, list_df_daily)
            df = rescale_batch(df, df_right, df_daily, topic_code, drop=drop)

    if add_daily_end:  # attempt to add daily data at the end
        daily_possible = [df_daily for df_daily in list_df_daily if df_daily.index.max() > df.index.max()]
        if len(daily_possible) != 0:
            column = df.columns[0]
            while len(daily_possible) > 0:
                candidate = max(daily_possible, key=lambda df_daily: df_daily.index.intersection(df.index))
                overlap_len = len(df.index.intersection(candidate.index))
                if overlap_len == 0:  # not possible to add the data since there is not overlap
                    break
                df = merge_trends_batches(df, candidate, overlap_len, column,
                                          is_hour=False, verbose=False, drop='right')
                df = df * 100 / df.max()
                daily_possible = [df_daily for df_daily in daily_possible if df_daily.index.max() > df.index.max()]
    return df


def find_min_dates_queries(list_dates: List[Tuple[datetime, datetime]], number: int) -> List[Tuple[datetime, datetime]]:
    """
    return the list of dates to query to form the minimum number of queries covering all dates provided
    uses largest_dates_iterator to determine if an interval can be queried
    :param list_dates: sorted list of tuples (begin, end) that can be queried
    :param number: number of queries that will be used to retrieve data on the interval. Used by largest_dates_iterator
    """
    if len(list_dates) == 0:
        return []
    root = (list_dates[0][0], list_dates[-1][1])
    if (root[1] - root[0]).days < max_query_days:
        return [root]
    else:
        # construct the tree
        class Node:
            def __init__(self, begin, end):
                self.begin = begin
                self.end = end
                largest_begin, largest_end = largest_dates_iterator(begin, end, number)
                self.feasible = (largest_end - largest_begin).days < max_query_days
                self.child = []
                self.parent = []
                self.covered = False

            def __str__(self):
                return str(self.begin.date()) + " " + str(self.end.date())

            def set_covered(self):
                self.covered = True
                # its 2 parents are by extension also covered
                if len(self.parent) > 0:
                    self.parent[0].set_covered()
                    self.parent[1].set_covered()

            def add_child(self, node):
                self.child.append(node)
                node.parent.append(self)

        def return_best_date(node: Node):
            queue = [node]
            dates = []
            while queue:
                node = queue.pop()
                if node.feasible and not node.covered:
                    node.set_covered()
                    dates.append((node.begin, node.end))
                elif not node.covered:
                    queue.append(node.parent[0])
                    queue.append(node.parent[1])
            return dates

        # construct the first nodes in the tree
        list_node = {0: [Node(a, b) for a, b in list_dates]}
        for depth in range(1, len(list_dates)):
            # add the child
            list_node[depth] = []
            for node_a, node_b in zip(list_node[depth-1], list_node[depth-1][1:]):
                node_cur = Node(node_a.begin, node_b.end)
                list_node[depth].append(node_cur)
                node_a.add_child(node_cur)
                node_b.add_child(node_cur)
        # retrieve the best interval by starting on the node at the largest depth
        best_dates = return_best_date(list_node[len(list_dates)-1][0])
        return best_dates


def min_timeframes_holes(list_df, number, overlap=30):
    """
    OLD VERSION
    gives the list of tuples of dates that must be queried to cover the holes between dataframes, each tuple having an
        overlap on the dataframe provided
    :param list_df: list of dataframe, having holes between them.
    :param number: of daily request made for the timeframes
    :param overlap: number of days that must be overlap between a dataframe and the date range provided
    """
    # list of tuple (begin, end) needed for the normalization
    list_interval = timeframe_normalize_clusters(list_df, overlap)
    for interval in list_interval:
        print(interval)
    max_day = timedelta(days=200)
    list_query = [list(dates_iterator(begin, end, number)) for begin, end in list_interval]
    dates_query = []
    max_iter = len(list_query)-2
    skip_next = False

    for i, (list_date_left, list_date_right) in enumerate(zip(list_query, list_query[1:])):
        min_date = min(list_date_left)[0]
        max_date = max(list_date_right, key=operator.itemgetter(1))[1]
        if skip_next:
            if i == max_iter:
                min_right = min(list_date_right)[0]
                dates_query.append((min_right, max_date))
            skip_next = False
            continue
        if max_date - min_date <= max_day:
            dates_query.append((min_date, max_date))
            skip_next = True
        else:
            max_left = max(list_date_left, key=operator.itemgetter(1))[1]
            dates_query.append((min_date, max_left))
            if i == max_iter:
                min_right = min(list_date_right)[0]
                dates_query.append((min_right, max_date))
    return dates_query


def collect_holes_data(topic_mid: str, topic_title: str, number: int, begin: datetime, end: datetime, geo: str,
                       cat: int = 0, verbose: bool = True) -> pd.DataFrame:
    """
    collect data using number daily requests and saves it to csv
    :param topic_mid: topic code
    :param topic_title: name of the topic
    :param number: number of queries to do
    :param begin: first day that the query must cover
    :param end: last day that the query must cover
    :param geo: geocode that must be queried
    :param cat: category to filter
    :param verbose: whether to print information while collecting the data or not
    :return df_daily: data collected using n_request
    """
    dir = "../data/trends/collect_gap/"
    timeframe_included = dates_to_timeframe(begin, end).replace(" ", "-")
    filename = f"{dir}{geo}-{topic_title}-{timeframe_included}.csv"
    df_daily = mean_query(number, begin, end, topic_mid, geo=geo, cat=cat, verbose=verbose)
    if verbose:
        print(f"daily requests saved to {filename}")
    df_daily.to_csv(filename)
    return df_daily


def scale_df(df, topic):
    """
    Return a list of the scaled df. If there is always an overlap, the list contains one df.
    Otherwhise, the list contains as many df as there are clusters of periods without missing data
    Each df has its first datetime beginning at 0h and its last datetime ending at 23h
    """
    batch_id = df["batch_id"].to_list()

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    batch_id = f7(batch_id)
    list_scaled_df = []
    scaled_df = pd.DataFrame()
    for i, j in enumerate(batch_id):
        if j < 0:  # the batch id was not valid
            if not scaled_df.empty:
                list_scaled_df.append(scaled_df)
            scaled_df = pd.DataFrame()
            continue

        batch_df = df[df["batch_id"] == j].drop(columns=["batch_id"])
        index_overlap = scaled_df.index.intersection(batch_df.index)
        overlap_hours = len(index_overlap)
        overlap_left = scaled_df.loc[index_overlap]
        overlap_right = batch_df.loc[index_overlap]
        if overlap_hours == 0 and scaled_df.empty:
            scaled_df = merge_trends_batches(scaled_df, batch_df, overlap_hours, topic)
        elif (overlap_left[topic] * overlap_right[topic]).sum() == 0:  # cannot perform the merge
            list_scaled_df.append(scaled_df)
            scaled_df = batch_df
        else:
            scaled_df = merge_trends_batches(scaled_df, batch_df, overlap_hours, topic)
    list_scaled_df.append(scaled_df)

    return list_scaled_df


def correct_batch_id(topic_title: str, topic_code: str, geo: str) -> pd.DataFrame:
    """
    correct the batch_id of the batch collected
    a batch_id must be negative if either
        - the data could not be collected on the given period
        - there is not less than 3 datapoints with value > 10
    :param topic_title: title of the topic to look at
    :param topic_code: code of the topic to look at
    :param geo: geocode of the dataframe to look at
    :return final_df: dataframe where the batch_id is set to negative if needed. This dataframe is saved in
        data/trends/collect as a csv file
    """
    data_dir = "../data/trends/collect"
    csv_file = f'{data_dir}/{geo}-{topic_title}.csv'
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_file, parse_dates=['date'], date_parser=date_parser).set_index('date')
    list_batches = df.batch_id.unique()

    final_df = pd.DataFrame()
    cur_batch_id = 0  # always positive
    previous_df_batch = None
    previous_batch_id = None

    for i, batch_id in enumerate(list_batches):
        df_batch = df[df['batch_id'] == batch_id]
        # a valid batch must have at least more than 3 values above 10
        if len(np.where(df_batch[topic_code] > 10)[0]) <= 3:
            df_batch = df_batch.assign(batch_id=-cur_batch_id)
            batch_id = -cur_batch_id
        else:  # the batch is valid and must have a positive batch id
            df_batch = df_batch.assign(batch_id=cur_batch_id)
            batch_id = cur_batch_id

        # there is a gap between this batch and the previous one and the two batches don't have a negative batch_id
        if (i != 0 and len(previous_df_batch.index.intersection(df_batch.index)) == 0
                and batch_id > 0 and previous_batch_id > 0):
            min_date = previous_df_batch.index.max()
            max_date = df_batch.index.min()
            days = pd.date_range(min_date, max_date, freq='H')
            data = np.zeros(len(days))
            batch_id_vector = [-cur_batch_id for i in range(len(days))]
            df_zero = pd.DataFrame({'date': days, topic_code: data, 'batch_id': batch_id_vector})
            df_zero = df_zero.set_index('date')
            cur_batch_id += 1  # count as if 2 batches were added
            df_batch = df_zero.append(df_batch)

        previous_df_batch = df_batch
        previous_batch_id = batch_id
        final_df = final_df.append(df_batch)
        cur_batch_id += 1
    final_df.to_csv(csv_file, index=True)
    return final_df


def actualize_trends(geocodes: Dict[str, str],
                     topics: Dict[str, str],
                     verbose: bool = True,
                     plot: bool = False,
                     only_hourly: bool = False,
                     refresh_daily: bool = True):
    """
    actualize the trends data in several steps:
    1) collect the hourly data
    2) correct the batch ids
    3) collect daily data on the gaps and form the final model data
    :param geocodes: dict of localizations
    :param topics: dict of topic_name: topic_code for trends
    :param verbose: whether to print information during the run or not
    :param plot: whether to plot the model data created or not
    :param only_hourly: whether to only actualize the hourly data or not
    :param refresh_daily: whether to refresh the daily data or not
    """
    # collect the hourly data
    for loc in geocodes:
        for topic_name, topic_code in topics.items():
            collect_historical_interest(topic_code, topic_name, loc, verbose=verbose)
    # correct the batch id
    if not only_hourly:
        for loc in geocodes:
            for topic_name, topic_code in topics.items():
                correct_batch_id(topic_name, topic_code, loc)
        # collect the data on the gaps
        collect_all_daily_gap(geocodes, topics, plot=plot, verbose=verbose, refresh=refresh_daily)


def actualize_trends_using_daily(geocodes: Dict[str, str],
                                 topics: Dict[str, str],
                                 verbose: bool = True,
                                 plot: bool = False,
                                 nb_mean: int = 10,
                                 overlap: int = 30,
                                 begin: datetime = None,
                                 refresh: bool = True):
    """
    actualize trends data using only daily requests and save it to the data/trends/model directory
    :param geocodes: dict of localizations
    :param topics: dict of topic_name: topic_code for trends
    :param verbose: whether to print information during the run or not
    :param plot: whether to plot the model data created or not
    :param overlap: number of days that must be overlapping
    :param nb_mean: number of mean requests to do to actualize the daily data
    :param refresh: whether to actually refresh the data or not. If False, generates model data based on the existing
        saved trends
    :param begin: first date where data is needed
    """
    if begin is None:
        begin = datetime.strptime("2020-01-01", "%Y-%m-%d")
    dir_daily = "../data/trends/collect_daily/"
    dir_model = "../data/trends/model/"
    plot_dir = "../plot/trends/"
    for loc in geocodes:
        for topic_name, topic_code in topics.items():
            filename = f"{dir_daily}{loc}-{topic_name}.csv"
            max_lag = np.floor(np.sqrt(nb_mean - 1))  # max lag used by dates iterator if every date can be queried
            length = max_query_days - 2 * max_lag  # length of the queries that will be done
            latest_day = latest_trends_daily_date()
            is_updated = False
            # load the saved csv file if it exist:
            if os.path.exists(filename):
                df_tot = pd.read_csv(filename, parse_dates=['date'], date_parser=date_parser_daily).set_index('date')
                # retrieve the dates already covered
                gb = df_tot.groupby("batch_id")
                df = gb.get_group(len(gb.size())-1)
                begin_covered = df.index.min().to_pydatetime()
                end_covered = df_tot.index.max().to_pydatetime()
                if end_covered == latest_day or not refresh:  # no need to update the file
                    is_updated = True
                else:
                    if (end_covered - begin_covered).days != length:  # the last batch registered was not the longest one
                        # remove the last batch as a longer can be queried
                        batch_id = df.iloc[0]["batch_id"]
                        df_tot = df_tot.loc[df_tot["batch_id"] != batch_id]
                        cur_begin = begin_covered
                        cur_end = cur_begin + timedelta(days=length)
                    else:  # the last batch was the best one that could be queried
                        batch_id = df_tot.iloc[-1]["batch_id"]
                        cur_begin = begin_covered + timedelta(days=(length - overlap))
                        cur_end = end_covered + timedelta(days=(length - overlap))
            else:  # no data existed previously
                if not refresh:  # cannot perform new queries if refresh == False
                    continue
                cur_begin = deepcopy(begin)
                cur_end = cur_begin + timedelta(days=length)
                df_tot = pd.DataFrame()
                batch_id = 0
            if not is_updated:
                list_dates = []
                last_length = max_query_days - overlap - nb_mean

                while cur_end < latest_day:
                    list_dates.append((cur_begin, cur_end))
                    cur_begin += timedelta(days=(length - overlap))
                    cur_end += timedelta(days=(length - overlap))

                cur_end = latest_day
                if (cur_end - cur_begin).days <= last_length:  # the last query can be done safely
                    list_dates.append((cur_begin, cur_end))
                else:  # need to split the last query into 2 sets of queries
                    cur_end = cur_begin + timedelta(max_query_days - nb_mean)
                    list_dates.append((cur_begin, cur_end))
                    cur_begin += timedelta(days=(length - overlap))
                    cur_end = latest_day
                    list_dates.append((cur_begin, cur_end))

                # send the queries
                for i, (date_from, date_to) in enumerate(list_dates):
                    if verbose:
                        print(f"{loc}-{topic_name}: retrieving batch {i+1}/{len(list_dates)} ")
                    df = mean_query(nb_mean, date_from, date_to, topic_code, loc, verbose=verbose)
                    df["batch_id"] = batch_id
                    df_tot = df_tot.append(df)
                    df_tot.to_csv(filename)
                    if verbose:
                        print(f"daily requests saved to {filename}")
                    batch_id += 1
            # construct the model data
            gb = df_tot.groupby("batch_id")
            list_df = [gb.get_group(x)[[topic_code]] for x in gb.groups]
            model = list_df[0]
            for df in list_df[1:]:
                model = merge_trends_batches(model, df, overlap + 1, topic_code, is_hour=False, verbose=False)
            model.to_csv(f"{dir_model}{loc}-{topic_name}.csv")
            if plot:
                plot_trends(model, topic_code, show=False)
                plt.title(f"{loc}: {topic_name}")
                plt.savefig(f"{plot_dir}{loc}-{topic_name}", facecolor='white')
                plt.show()


def collect_all_daily_gap(geocodes: Dict[str, str], topics: Dict[str, str], plot=False, verbose=True, refresh=True):
    """
    collect daily data on the gaps on all geocodes and topics specified
    :param geocodes: dict of geocodes to actualize
    :param topics: dict of topics to actualize
    :param plot: whether to plot the final model data or not
    :param verbose: whether to print information during the collection of data or not
    :param refresh: whether to refresh the daily data or not
    """
    plot_dir = "../plot/trends"
    for geo in geocodes:
        for topic_title, topic_mid in topics.items():
            if verbose:
                print(f"{geo}: {topic_title}")
            model = daily_gap_and_model_data(geo, topic_title, topic_mid, 20, overlap=30, verbose=verbose,
                                             refresh=refresh)
            if plot:
                plot_trends(model, topic_mid, show=False)
                plt.title(f"{geo}: {topic_title}")
                plt.savefig(f"{plot_dir}/{geo}-{topic_title}", facecolor='white')
                plt.show()


if __name__ == "__main__":
    """
    #actualize_trends(extract_topics(), start_month=3)
    #unscaled, scaled = get_historical_interest_normalized('/m/0cjf0', "2020-02-01", "2020-10-28", geo='BE',
    #                                                      sleep_fun=lambda: 60 + 10 * random.random())
    actualize_hospi()
    actualize_github()
    """

    """

    geocodes = {
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

    data_hourly_dir = "../data/trends/collect"
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    for geo, geo_name in geocodes.items():
        df_hourly = {name: pd.read_csv(f"{data_hourly_dir}/{geo}-{name}.csv", parse_dates=['date'],
                                       date_parser=date_parser).set_index('date') for name in list_topics}
        for topic_title, topic_code in list_topics.items():
            print(f" ----- {geo}:{topic_title} -----")
            scaled_df = scale_df(df_hourly[topic_title], topic_code)
            list_timeframe = min_timeframes_holes(scaled_df, 20)
            for begin, end in list_timeframe:
                collect_holes_data(topic_code, topic_title, 20, begin, end, geo, verbose=True)

    
    list_topics = {
        'Fièvre': '/m/0cjf0',
        'Mal de gorge': '/m/0b76bty',
        'Dyspnée': '/m/01cdt5',
        'Agueusie': '/m/05sfr2',
        'Anosmie': '/m/0m7pl',
        'Virus': '/m/0g9pc',
        'Épidémie': '/m/0hn9s',
        'Symptôme': '/m/01b_06',
        'Thermomètre': '/m/07mf1',
        'Grippe espagnole': '/m/01c751',
        'Paracétamol': '/m/0lbt3',
        'Respiration': '/m/02gy9_',
        'Toux': '/m/01b_21',
        'Coronavirus': '/m/01cpyy'
    }

    geocodes = {
        'FR-A': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-B': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-C': "Auvergne-Rhône-Alpes",
        'FR-D': "Bourgogne-Franche-Comté",
        'FR-E': 'Bretagne',
        'FR-F': 'Centre-Val de Loire',
        'FR-G': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-H': 'Corse',
        'FR-I': "Bourgogne-Franche-Comté",
        'FR-J': 'Ile-de-France',
        'FR-K': 'Languedoc-Roussillon-Midi-Pyrénées',
        'FR-L': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-M': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-N': 'Languedoc-Roussillon-Midi-Pyrénées',
        'FR-O': 'Nord-Pas-de-Calais-Picardie',
        'FR-P': "Normandie",
        'FR-Q': "Normandie",
        'FR-R': 'Pays de la Loire',
        'FR-S': 'Nord-Pas-de-Calais-Picardie',
        'FR-T': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-U': "Provence-Alpes-Côte d'Azur",
        'FR-V': "Auvergne-Rhône-Alpes",
        'BE': "Belgique"
    }

    for title, mid in list_topics.items():
        collect_historical_interest(mid, title, geo='FR-T')
    """
    list_topics = {
        'Fièvre': '/m/0cjf0',
        'Mal de gorge': '/m/0b76bty',
        #'Dyspnée': '/m/01cdt5',
        #'Agueusie': '/m/05sfr2',
        #'Anosmie': '/m/0m7pl',
        #'Virus': '/m/0g9pc',
        #'Épidémie': '/m/0hn9s',
        'Symptôme': '/m/01b_06',
        #'Thermomètre': '/m/07mf1',
        #'Grippe espagnole': '/m/01c751',
        #'Paracétamol': '/m/0lbt3',
        #'Respiration': '/m/02gy9_',
        #'Toux': '/m/01b_21',
        #'Coronavirus': '/m/01cpyy'
    }

    geo = {
        'FR-A': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-B': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-C': "Auvergne-Rhône-Alpes",
        'FR-D': "Bourgogne-Franche-Comté",
        'FR-E': 'Bretagne',
        'FR-F': 'Centre-Val de Loire',
        'FR-G': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-H': 'Corse',
        'FR-I': "Bourgogne-Franche-Comté",
        'FR-J': 'Ile-de-France',
        'FR-K': 'Languedoc-Roussillon-Midi-Pyrénées',
        'FR-L': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-M': "Alsace-Champagne-Ardenne-Lorraine",
        'FR-N': 'Languedoc-Roussillon-Midi-Pyrénées',
        'FR-O': 'Nord-Pas-de-Calais-Picardie',
        'FR-P': "Normandie",
        'FR-Q': "Normandie",
        'FR-R': 'Pays de la Loire',
        'FR-S': 'Nord-Pas-de-Calais-Picardie',
        'FR-T': "Aquitaine-Limousin-Poitou-Charentes",
        'FR-U': "Provence-Alpes-Côte d'Azur",
        'FR-V': "Auvergne-Rhône-Alpes",
        'BE': "Belgique"
    }
    #actualize_trends(geo, list_topics, plot=True, only_hourly=True, refresh_daily=False)
    actualize_hospi()
    actualize_trends_using_daily(util.european_geocodes, list_topics, plot=False, refresh=True)
