from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import spearmanr


def extract_topic(filename="topics.txt"):
    """
    extract the pairs of "topics_url description" in the file provided
    :param filename: file were the topics are written. Each valid line must be in the format "topic_url topic_name"
        invalid lines are ignored
    :return: dictionary of {topic: url} for each topic provided
    """
    results = {}
    pattern = "(\S+)\s(.+)"
    with open(filename) as file:
        for line in file:
            search_obj = re.match(pattern, line)
            if search_obj is not None:
                results[search_obj.group(1)] = search_obj.group(2)
    return results


def extract_term(filename="symptoms.txt"):
    """
    extract the search terms in the file provided
    :param filename: file where the terms are written. Each line corresponds exactly to one term. Empty lines are ignored
    :return: list of terms in the file
    """
    results = []
    pattern = "\S"
    with open(filename) as file:
        for line in file:
            if re.match(pattern, line) is not None:
                results.append(line)
    return results


def trends_to_csv(topics,
                  output_filename='search_trends.csv',
                  timeframe="2020-03-11 " + datetime.today().strftime('%Y-%m-%d')):
    """
    analyze multiple trends over the timeframe provided. Each trends is computed individually => can be time consuming
    :param output_filename: CSV filename where to write the results
    :param topics: list of search items or pair of "topic_url topic_name" to be searched
    :param timeframe: time period to search. Default to 11 march until today
    :return:
    inspired by a tutorial found on https://www.honchosearch.com/blog/seo/how-to-use-python-pytrends-to-automate-google-trends-data/
    """
    dataset = []
    pytrends = TrendReq(hl='en-US', tz=360)
    for search in topics:
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
    :param hospitals: CSV file with the number of hospitalization
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
    pd.DataFrame(data, columns=['Term', "Correlation", 'Pvalue']).to_csv(output_file)


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
    pytrends = TrendReq(hl='en-US', tz=360)
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


if __name__ == "__main__":
    #plot_interest_over_time(extract_topic())
    #trends_to_csv(extract_term())
    spearman_hospitalization('search_trends.csv', 'hospitalization.csv')
