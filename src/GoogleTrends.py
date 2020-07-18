import difflib

from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr
from random import random
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge


def extract_topics(filename="topics.txt", toList=False):
    """
    Extracts the pairs of "topics_mid topic_title" in the file provided
    :param filename: file were the topics are written. Can be both a filename or a list of files
        Each valid line must be in the format "topic_mid topic_title"
    :param toList: whether to return the list of keys or the whole dictionary
    :return: dictionary of {topic_title: topic_mid} for each topic provided
    """
    results = {}
    pattern = "(\S+)\s(.+)"
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


def extract_queries(filename="symptoms.txt"):
    """
    Extracts the search terms in the file provided
    :param filename: file where the terms are written. Can be both a filename or a list of files
        Each line corresponds exactly to one term
    :return: list of terms in the file
    """
    results = []
    if isinstance(filename, str):
        filename2 = [filename]
    else:
        filename2 = filename
    for name in filename2:
        with open(name) as file:
            results += file.read().splitlines()
    return results


def trends_to_csv(topics,
                  output_filename='search_trends.csv',
                  timeframe="2020-03-15 " + datetime.today().strftime('%Y-%m-%d')):
    """
    Analyses multiple trends over the timeframe provided. Each trends is computed individually => can be time consuming
    :param output_filename: CSV filename where to write the results
    :param topics: list of search items or pair of "topic_mid topic_title" to be searched
    :param timeframe: time period to search. Default to 11 march until today
    :return:
    inspired by a tutorial found on https://www.honchosearch.com/blog/seo/how-to-use-python-pytrends-to-automate-google-trends-data/
    """
    dataset = []
    pytrends = TrendReq(timeout=(100, 250))
    for search in topics:
        time.sleep(2 + random() * 2)
        pytrends.build_payload([search], cat=0, timeframe=timeframe, geo='BE', gprop='')
        data = pytrends.interest_over_time()
        if not data.empty:
            data = data.drop(labels=['isPartial'], axis='columns')
            dataset.append(data)
    result = pd.concat(dataset, axis=1)
    result.index.rename('DATE', inplace=True)
    result.to_csv(output_filename)


def offset_trends_hospi(n, indexes_file="trends_2.csv", hospitals_file="hospitalization.csv", begin=None,
                        end=None):
    """
    extract the hospitalization during [begin, end] and the trends from [begin - n days, end]
    an exception is raised if the dates provided are invalid
    :param indexes_file: CSV file with the indexes for each topic
    :param hospitals_file: CSV file with the number of hospitalization
    :param n: number of days in the offset
    :param begin: date of first hospitalization. None = first date in the hospitalization file
    :param end: date of last hospitalization. None = last date in the hospitalization file
    :return: df, hospi
        * df: index dataframe, from begin date minus n days to end date
        * hospi: hospitalizations cases, from begin date to end date
    """
    if len(indexes_file.split('.')) != 2:
        indexes_file += '.csv'
    if len(hospitals_file.split('.')) != 2:
        hospitals_file += '.csv'
    hospitals = pd.read_csv(hospitals_file)
    date_max_hospi = hospitals['DATE'].max()
    indexes = pd.read_csv(indexes_file)
    if begin is None:
        begin = hospitals['DATE'].min()
    elif hospitals['DATE'].min() > begin:
        raise Exception("offset_hospi_trends: not enough data in hospital file: "
                        "first date is {:10s} but asked data from {:10s}".format(hospitals['DATE'].min(), begin))
    if end is None:
        end = hospitals['DATE'].max()
    begin_trends = str(date.fromisoformat(begin) - timedelta(days=n))
    end_trends = str(date.fromisoformat(end) - timedelta(days=n))
    if indexes['DATE'].min() > begin_trends:
        raise Exception("offset_hospi_trends: not enough data in indexes file: "
                        "first date is {:10s} but asked data from {:10s}".format(indexes['DATE'].min(), begin_trends))
    elif str(date.fromisoformat(end) - timedelta(n+1)) > indexes['DATE'].max():
        raise Exception("offset_hospi_trends: not enough data in hospital file: "
                        "last date is {:10s} but asked data from {:10s}".format(hospitals['DATE'].max(), str(date.fromisoformat(end) - timedelta(n+1))))
    hospi = hospitals[hospitals['DATE'].between(begin, end)].groupby(['DATE']).agg({'NEW_IN': 'sum'})['NEW_IN'].tolist()
    indexes = indexes[indexes['DATE'].between(begin_trends, end_trends)]
    return indexes, hospi, date_max_hospi


def spearman_hospitalization(indexes_file, hospitals_file, output_file="correlation.csv"):
    """
    Performs a spearman correlation test between a query index over time and the number of cases over time
    :param indexes_file: CSV file with the indexes for each topic
    :param hospitals_file: CSV file with the number of hospitalization
    :param output_file: file where to write the coefficients and p-value
    :return: None. Write the tests results to the file provided
    """
    indexes = pd.read_csv(indexes_file)
    hospitals = pd.read_csv(hospitals_file)
    date_min = max(indexes['DATE'].min(), hospitals['DATE'].min())  # get a common time interval for both files
    date_max = min(indexes['DATE'].max(), hospitals['DATE'].max())
    indexes = indexes[indexes['DATE'].between(date_min, date_max)]
    hospitals = hospitals[hospitals['DATE'].between(date_min, date_max)].groupby(['DATE']).agg({'NEW_IN': 'sum'})
    data = []
    for row in indexes:
        if row == 'DATE':
            continue
        coef, p = spearmanr(indexes[row], hospitals['NEW_IN'])
        data.append([row, coef, p])
    df = pd.DataFrame(data, columns=['Term', "Correlation", 'Pvalue'])
    df.reindex(df["Correlation"].abs().sort_values().index).to_csv(output_file)


def plot_interest_over_time(topics,
                            timeframe="2020-03-15 " + datetime.today().strftime('%Y-%m-%d'),
                            geo="BE",
                            title="Interest over time in Belgium"):
    """
    Plots the evolution over the given period of different topics in Belgium
    :param topics: dictionary of topics / list of terms to be searched. Up to length of five items
    :param timeframe: time period to search. Default to 11 march until today
    :param geo: geographic region
    :param title: title for the graph
    :return: None. Plot the graph of interest over time
    """
    assert 5 >= len(topics) > 0, "maximum five items can be provided"
    pytrends = TrendReq()
    if isinstance(topics, dict):
        kw_list = topics.keys()
    else:
        kw_list = topics
    pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop='')
    interest_over_time_df = pytrends.interest_over_time()
    sns.set(color_codes=True)
    dx = interest_over_time_df.plot.line(figsize=(8, 5), title=title)
    if isinstance(topics, dict):
        dx.legend(topics.values())
    dx.set_xlabel('Date')
    dx.set_ylim([0, 100])
    dx.set_ylabel('Trends index')
    plt.show()


def write_related_topics(keywords, filename="topics_generated.txt",
                         timeframe="2020-03-15 " + datetime.today().strftime('%Y-%m-%d'),
                         geo='BE'):
    """
    Writes the topics related to the topics provided into the filename. Duplicates are removed
    :param keywords: dictionary of topics or list of terms
    :param filename: file where to write the topics
    :param timeframe: time period to search. Default to 11 march until today
    :param geo: geographic region
    :return: None. Write the pairs of topic_mid topic_title in the file
    """
    if isinstance(keywords, dict):
        reference = keywords.keys()
    else:
        reference = keywords
    data = pd.DataFrame()
    pytrends = TrendReq(timeout=(10, 25))
    for item in reference:
        time.sleep(2 + random() * 2)
        pytrends.build_payload([item], cat=0, timeframe=timeframe, geo=geo, gprop='')
        req = pytrends.related_topics()
        # append the related topics (both rising and top topics) to the dataset
        if req[item]['rising'] is None or req[item]['rising'].empty:
            if req[item]['top'] is not None and not req[item]['top'].empty:
                data = pd.concat([data, req[item]['top'][['topic_mid', 'topic_title']]])
        else:
            if req[item]['top'] is None or req[item]['top'].empty:
                data = pd.concat([data, req[item]['rising'][['topic_mid', 'topic_title']]])
            else:
                data = pd.concat([data, req[item]['rising'][['topic_mid', 'topic_title']],
                                  req[item]['top'][['topic_mid', 'topic_title']]])
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    with open(filename, "w") as file:
        for mid, title in data.values:
            file.write(mid + " " + title + "\n")


def write_related_queries(keywords, filename="queries_generated.txt",
                          timeframe="2020-03-15 " + datetime.today().strftime('%Y-%m-%d'),
                          geo='BE'):
    """
    Writes the queries related to the keywords provided into the filename. Duplicates are removed
    :param keywords: dictionary of topics or list of terms
    :param filename: file where to write the related queries
    :param timeframe: time period to search. Default to 11 march until today
    :param geo: geographic region
    :return: None. Write the queries in the file
    """
    if isinstance(keywords, dict):
        reference = keywords.keys()
    else:
        reference = keywords
    data = pd.DataFrame()
    pytrends = TrendReq(timeout=(10, 25))
    for item in reference:
        time.sleep(2 + random() * 2)
        pytrends.build_payload([item], cat=0, timeframe=timeframe, geo=geo, gprop='')
        req = pytrends.related_queries()
        # append the related queries (both rising and top topics) to the dataset
        if req[item]['rising'] is None or req[item]['rising'].empty:
            if req[item]['top'] is not None and not req[item]['top'].empty:
                data = pd.concat([data, req[item]['top']['query']])
        else:
            if req[item]['top'] is None or req[item]['top'].empty:
                data = pd.concat([data, req[item]['rising']['query']])
            else:
                data = pd.concat([data, req[item]['rising']['query'], req[item]['top']['query']])
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    with open(filename, "w") as file:
        for [title] in data.values:
            file.write(title + "\n")


def prediction_hospitalizations(correlation_file, n, days, abs_val=True, date_begin='2020-03-15', date_end='2020-07-09'):
    """
    Finds the best classifier in order to predict the number of hospitalisations based on data of previous days
    (one-day prediction)
    :param indexes_file: CSV file with the indexes for each topic
    :param hospitals_file: CSV file with the number of hospitalization
    :param correlation_file: CSV file with the correlation between terms
    :param n: Number of most correlated topics or queries to consider
    :param days: Number of days we want to predict
    :param abs_val: Boolean that indicates if the number of hospitalisations should be an absolute value (True) or a relative one (False)
    :param date_begin: First date for which we have the number of new hospitalisations
    :param date_end: Last date that we use in order to predict the number of new hospitalisations
    :return: The best prediction for the number of hospitalisation of tomorrow
    """
    # Get dates and number of new hospitalizations
    date_begin = str(date.fromisoformat(date_begin) + timedelta(days))
    trends, hospi, date_max_hospi = offset_trends_hospi(days, begin=date_begin, end=date_end)
    dates = trends['DATE'].tolist()

    diff = len(dates) - len(hospi)
    dates = dates[diff:]
    prediction_known = True
    if date_end > date_max_hospi:
        prediction_known = False

    # Transform number of new hospitalisations into indexes if abs_val = False
    if abs_val:
        data = {'Date': dates, 'New_hospitalisations': hospi}
    else:
        max_hosp = max(hospi)
        new_hosp = []
        for hosp in hospi:
            new_hosp.append((hosp / max_hosp) * 100)
        data = {'Date': dates, 'New_hospitalisations': new_hosp}
    df = pd.DataFrame(data)

    # Get the n most correlated topics and queries
    terms, queries = filter_correlated(correlation_file, n)

    # Create dataframe with most correlated words
    loc = 1
    for term in terms:
        searches = trends[term].tolist()[diff:]
        df.insert(loc, queries[loc - 1], searches, True)
        loc += 1

    # Prediction of hospitalizations
    df['Date'] = pd.to_datetime(df['Date']).map(datetime.toordinal)
    train = df.head(int(len(df) - 1))
    X_train = train[train.columns.difference(['New_hospitalisations'])]
    y_train = train['New_hospitalisations']
    y_train = y_train.head(int(len(df) - 1))
    test = df.tail(1)
    X_test = test[test.columns.difference(['New_hospitalisations'])]
    y_test = test['New_hospitalisations']
    y_test = y_test.tolist()
    #print(df)

    # The classification model that we will use has to be a regression since the result should not be binary
    classifier = prediction_models(X_train, y_train, X_test, y_test, prediction_known=prediction_known)
    if prediction_known:
        print("Estimation of the number of new hospi ({:10s}) using data from {:d} day{:s} before = {:.3f}"
            .format(date_end, days, 's' if days > 1 else '',classifier[0]))
    else:
        print("These are the predictions for ({:10s}) using data from {:d} day{:s} before"
              .format(date_end, days, 's' if days > 1 else ''))
    return classifier


def prediction_models(X_train, y_train, X_test, y_test, prediction_known=True):
    """
    Finds the best classifier in order to predict the number of new hospitalisations
    :param X_train: features used for learning by the model
    :param y_train: number of new hospitalisations known
    :param X_test: all features except the hospitalisation that will be used for the prediction
    :param y_test: true number of new hospitalisations (the prediction should be close to this value)
    :return: The value of the best prediction for the number of new hospitalisations
    """
    difference = float("inf")
    classifier = 0
    if prediction_known:
        print("Trying to predict {:.3f} ".format(y_test[0]))
    else:
        print("Trying to predict a value in the future")

    # Decision Tree Regression
    model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_diff = abs(y_pred - y_test).tolist()[0]
    if new_diff < difference:
        difference = abs(y_pred - y_test).tolist()[0]
        classifier = y_pred
    error_percentage = (new_diff / y_test[0]) * 100
    if prediction_known:
        print("\t{:28s} predicted: {:7.3f}\t difference: {:+8.3f}\t ({:6.2f}% error)"
            .format("Decision Tree Regressor:", y_pred[0], (y_pred - y_test).tolist()[0], error_percentage))
    else:
        print("\t{:28s} predicted: {:7.3f}".format("Decision Tree Regressor:", y_pred[0]))

    # K nearest neighbors regressor
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_diff = abs(y_pred - y_test).tolist()[0]
    if new_diff < difference:
        difference = abs(y_pred - y_test).tolist()[0]
        classifier = y_pred
    error_percentage = (new_diff / y_test[0]) * 100
    if prediction_known:
        print("\t{:28s} predicted: {:7.3f}\t difference: {:+8.3f}\t ({:6.2f}% error)"
            .format("K-nearest neighbors Regressor:", y_pred[0], (y_pred - y_test).tolist()[0], error_percentage))
    else:
        print("\t{:28s} predicted: {:7.3f}".format("K-nearest neighbors Regressor:", y_pred[0]))

    # Random Forest Regression
    model = RandomForestRegressor(max_depth=8, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_diff = abs(y_pred - y_test).tolist()[0]
    if new_diff < difference:
        difference = abs(y_pred - y_test).tolist()[0]
        classifier = y_pred
    error_percentage = (new_diff / y_test[0]) * 100
    if prediction_known:
        print("\t{:28s} predicted: {:7.3f}\t difference: {:+8.3f}\t ({:6.2f}% error)"
            .format("Random Forest Regressor:", y_pred[0], (y_pred - y_test).tolist()[0], error_percentage))
    else:
        print("\t{:28s} predicted: {:7.3f}".format("Random Forest Regressor:", y_pred[0]))


    # Gradient Boosting Regression
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_diff = abs(y_pred - y_test).tolist()[0]
    if new_diff < difference:
        difference = abs(y_pred - y_test).tolist()[0]
        classifier = y_pred
    error_percentage = (new_diff / y_test[0]) * 100
    if prediction_known:
        print("\t{:28s} predicted: {:7.3f}\t difference: {:+8.3f}\t ({:6.2f}% error)"
            .format("Gradient Boosting Regressor:", y_pred[0], (y_pred - y_test).tolist()[0], error_percentage))
    else:
        print("\t{:28s} predicted: {:7.3f}".format("Gradient Boosting Regressor:", y_pred[0]))

    return classifier


def filter_correlated(correlation_file, n):
    """
    Finds the n most correlated and different topics or queries
    :param correlation_file: CSV file with the indexes for each topic
    :param n: Number of most correlated topics or queries to consider
    :return: list containing the n most correlated queries and list containing the n most correlated topics
    """
    correlation_df = pd.read_csv(correlation_file)
    correlation_df = correlation_df.fillna(value="HelloWorld")
    topics = list(reversed(correlation_df['Topic'].tolist()))
    terms = list(reversed(correlation_df['Term'].tolist()))
    final_list_topics = []
    final_list_terms = []

    while len(final_list_topics) < n:
        if topics[0] == "HelloWorld":
            topics[0] = terms[0]
        if len(final_list_topics) == 0:
            final_list_topics.append(topics[0])
            final_list_terms.append(terms[0])
        else:
            too_similar = False
            for item in final_list_topics:
                similarity = difflib.SequenceMatcher(None, item.lower(), topics[0].lower()).ratio()
                if similarity > 0.5 or item.lower() in topics[0].lower() or topics[0].lower() in item.lower():
                    too_similar = True
            if not too_similar:
                final_list_topics.append(topics[0])
                final_list_terms.append(terms[0])
        topics.remove(topics[0])
        terms.remove(terms[0])
    return final_list_terms, final_list_topics


def find_correlated(cases, queries: list = None, topics: dict = None, correlation_limit=0.65, max_iter=1,
                    timeframe="2020-03-15 " + datetime.today().strftime('%Y-%m-%d'),
                    geo='BE',
                    trends_file="trends_2.csv",
                    correlation_file="correlation_3.csv",
                    proxies='',
                    cat=0,
                    print_info=True):
    """
    find new correlated topics and queries with respect to the cases provided
    :param queries: initial queries. Related queries & topics will be retrieved from here
    :param topics: initial topics. Related queries & topics will be retrieved from here
    :param cases: data vector to be used. len(cases) == number of days in timeframe
    :param correlation_limit: absolute correlation used to explore the data for the next step. Correlation under this
        threshold will not be used to find new keywords
    :param max_iter: number of iteration used. Each iter add more keywords to the initial set, who will be used to
        get more and more keywords over the iterations
    :param timeframe: time period to search. Default to 15 march until 9 july
    :param geo: geographic region
    :param correlation_file: where to write the correlation and p value
    :param trends_file: where to write the RSV
    :param proxies: list of HTTPS proxies to use. '' means no proxy
    :param cat: category for the search. 0 means all categories, 45 is Health etc. List of categories is available at
        https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    :param print_info: True if information is supposed to be written during the run
    :return:
    """
    # get the related topics and queries and keep the most correlated with respect to the data provided
    assert queries is not None or topics is not None
    if len(trends_file.split('.')) != 2:
        trends_file += '.csv'
    if len(correlation_file.split('.')) != 2:
        correlation_file += '.csv'

    pytrends = TrendReq(hl='en-US', timeout=(100, 250), retries=2, backoff_factor=0.1, proxies=proxies)
    correlation_header = pd.DataFrame(columns=['Term', 'Topic', 'Correlation', 'Pvalue'])
    correlation_df = correlation_header.copy(True)
    correlation_batch = correlation_header.copy(True)
    interest_df = pd.DataFrame()
    keywords_used = set()  # keywords already used
    keywords_new = set()  # new keywords to use
    if queries is not None:
        keywords_new.update(queries)
        correlation_batch = pd.concat([correlation_batch, pd.DataFrame({'Term': queries})])
    if topics is not None:
        keywords_new.update(topics.keys())
        correlation_batch = pd.concat(
            [correlation_batch, pd.DataFrame({'Term': list(topics.keys()), 'Topic': list(topics.values())})])

    if print_info:
        init = time.perf_counter()

    for iteration in range(max_iter + 1):  # first iteration = initial keywords provided
        if print_info:
            iter_init = time.perf_counter()
            step_init = iter_init
            print("iteration {:d}/{:d}".format(iteration + 1, max_iter + 1) +
                  "\n\tadding related keywords and topics of {:d} keywords... ".format(len(keywords_new)), end="",
                  flush=True)
        if iteration != 0:
            correlation_batch = correlation_header.copy(True)  # batch of keywords for each iteration
            for word in keywords_new:  # add the new keywords
                time.sleep(random())
                pytrends.build_payload([word], cat=cat, timeframe=timeframe, geo=geo, gprop='')
                for i in range(2):
                    if i == 0:  # related queries
                        req = pytrends.related_queries()
                        description = ['query']
                        renaming = {description[0]: 'Term'}
                    else:  # related topics
                        req = pytrends.related_topics()
                        description = ['topic_mid', 'topic_title']
                        renaming = {description[0]: "Term", description[1]: "Topic"}
                    if req[word]['rising'] is None or req[word]['rising'].empty:
                        if req[word]['top'] is not None and not req[word]['top'].empty:
                            correlation_tmp = req[word]['top'][description]
                            correlation_batch = pd.concat([correlation_batch,
                                                           correlation_tmp.rename(columns=renaming).merge(
                                                               correlation_header, how='outer')])
                    else:
                        if req[word]['top'] is None or req[word]['top'].empty:
                            correlation_tmp = req[word]['rising'][description]
                        else:
                            correlation_tmp = pd.concat(
                                [req[word]['rising'][description], req[word]['top'][description]])
                        correlation_batch = pd.concat([correlation_batch,
                                                       correlation_tmp.rename(columns=renaming).merge(
                                                           correlation_header, how='outer')])
        correlation_batch.drop_duplicates(inplace=True)
        correlation_batch.reset_index(drop=True, inplace=True)

        # get the interest over time for the new keywords
        if print_info:
            print("Ended in {:.2f} s\n\tcomputing interest over time... ".format(time.perf_counter() - step_init),
                  end="", flush=True)
            step_init = time.perf_counter()
        interest = []
        for word in correlation_batch['Term']:
            if word in keywords_used:
                continue
            else:
                keywords_used.add(word)
            pytrends.build_payload([word], cat=cat, timeframe=timeframe, geo=geo, gprop='')
            interest_tmp = pytrends.interest_over_time()
            if not interest_tmp.empty:
                interest.append(interest_tmp.drop(labels=['isPartial'], axis='columns'))
        interest_batch = pd.concat(interest, axis=1)
        interest_batch.index.rename('DATE', inplace=True)

        # correlation test for each new keyword in the batch
        if print_info:
            print("Ended in {:.2f} s\n\tperforming correlation test... ".format(time.perf_counter() - step_init),
                  end="", flush=True)
            step_init = time.perf_counter()
        keywords_new = set()
        for row in interest_batch:
            if row == 'DATE':
                continue
            coef, p = spearmanr(interest_batch[row], cases)
            if abs(coef) >= correlation_limit or iteration == 0:  # only add relevant new keywords
                keywords_new.add(row)
            correlation_batch.loc[correlation_batch['Term'] == row, ['Correlation', 'Pvalue']] = coef, p

        # append the batches
        correlation_df = pd.concat([correlation_df, correlation_batch])
        if iteration == 0:
            interest_df = interest_batch.copy(True)
        else:
            interest_df = interest_df.join(interest_batch)
        if print_info:
            cur = time.perf_counter()
            print("Ended in {:.2f} s\n\t\ttotal iteration time: {:.2f} s\n\t\ttotal execution time: {:.2f} s"
                  .format(cur - step_init, cur - iter_init, cur - init), flush=True)

    interest_df.index.rename('DATE', inplace=True)  # same name as hospitalization file
    interest_df.to_csv(trends_file)
    correlation_df.drop_duplicates(inplace=True)
    correlation_df.reindex(correlation_df['Correlation'].abs().sort_values().index).to_csv(correlation_file,
                                                                                           index=False)
    return correlation_df, interest_df


def hospitalization_vector(hospitals_file):
    """
    Gets the number of cases per day in the file provided
    :param hospitals_file: CSV file with the number of hospitalization
    :return: list of cases for each day
    """
    hospitals = pd.read_csv(hospitals_file)
    hospitals = hospitals.groupby(['DATE']).agg({'NEW_IN': 'sum'})
    return hospitals.sort_values('DATE')['NEW_IN'].to_list()


if __name__ == "__main__":
    # plot_interest_over_time(extract_topics())
    # write_related_queries(extract_topics(toList=True) + extract_queries())
    # write_related_topics(extract_topics(toList=True) + extract_queries())
    # trends_to_csv(extract_queries(["queries_generated.txt", "symptoms.txt"]) + extract_topics(["topics_generated.txt", "topics.txt"], True))
    # spearman_hospitalization('search_trends.csv', 'hospitalization.csv')
    """cases = hospitalization_vector("hospitalization.csv")
    queries = extract_queries()
    topics = extract_topics()
    find_correlated(cases, queries=queries, topics=topics, max_iter=1)"""
    prediction_hospitalizations('correlation_3.csv', n=5, days=0, abs_val=True, date_begin='2020-03-15', date_end='2020-07-13')
    prediction_hospitalizations('correlation_3.csv', n=5, days=1, abs_val=True, date_begin='2020-03-15', date_end='2020-07-13')
    prediction_hospitalizations('correlation_3.csv', n=5, days=2, abs_val=True, date_begin='2020-03-15', date_end='2020-07-13')
    prediction_hospitalizations('correlation_3.csv', n=5, days=3, abs_val=True, date_begin='2020-03-15', date_end='2020-07-13')
    prediction_hospitalizations('correlation_3.csv', n=5, days=4, abs_val=True, date_begin='2020-03-15', date_end='2020-07-13')

