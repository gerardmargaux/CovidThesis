from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import spearmanr
from random import random
from time import sleep


def extract_topics(filename="topics.txt", toList=False):
    """
    extract the pairs of "topics_mid topic_title" in the file provided
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
    extract the search terms in the file provided
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
                  timeframe="2020-03-11 " + datetime.today().strftime('%Y-%m-%d')):
    """
    analyze multiple trends over the timeframe provided. Each trends is computed individually => can be time consuming
    :param output_filename: CSV filename where to write the results
    :param topics: list of search items or pair of "topic_mid topic_title" to be searched
    :param timeframe: time period to search. Default to 11 march until today
    :return:
    inspired by a tutorial found on https://www.honchosearch.com/blog/seo/how-to-use-python-pytrends-to-automate-google-trends-data/
    """
    dataset = []
    pytrends = TrendReq(timeout=(100, 250))
    for search in topics:
        sleep(2 + random() * 2)
        pytrends.build_payload([search], cat=0, timeframe=timeframe, geo='BE', gprop='')
        data = pytrends.interest_over_time()
        if not data.empty:
            data = data.drop(labels=['isPartial'], axis='columns')
            dataset.append(data)
    result = pd.concat(dataset, axis=1)
    result.index.rename('DATE', inplace=True)
    result.to_csv(output_filename)


def spearman_hospitalization(indexes_file, hospitals_file, output_file="correlation.csv"):
    """
    perform a spearman correlation test between a query index over time and the number of cases over time
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
                            timeframe="2020-03-11 " + datetime.today().strftime('%Y-%m-%d'),
                            geo="BE",
                            title="Interest over time in Belgium"):
    """
    plot the evolution over the given period of different topics in Belgium
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
                         timeframe="2020-03-11 " + datetime.today().strftime('%Y-%m-%d'),
                         geo='BE'):
    """
    write the topics related to the topics provided into the filename. Duplicates are removed
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
        sleep(2 + random() * 2)
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
                          timeframe="2020-03-11 " + datetime.today().strftime('%Y-%m-%d'),
                          geo='BE'):
    """
    write the queries related to the keywords provided into the filename. Duplicates are removed
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
        sleep(2 + random() * 2)
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


if __name__ == "__main__":
    # plot_interest_over_time(extract_topic())
    # write_related_queries(extract_topics(toList=True) + extract_queries())
    # write_related_topics(extract_topics(toList=True) + extract_queries())
    # trends_to_csv(extract_queries(["queries_generated.txt", "symptoms.txt"]) + extract_topics(["topics_generated.txt", "topics.txt"], True))
    spearman_hospitalization('search_trends.csv', 'hospitalization.csv')
