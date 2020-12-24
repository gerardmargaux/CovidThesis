import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import datetime
from time import sleep

# drop one batch

topics = {
    "Symptôme": "/m/01b_06"
}
topic_used = "Symptôme"
topic_code = topics[topic_used]
data_hourly_dir = "../data/trends/collect"
geo = "BE"
date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
df_hourly = {name: pd.read_csv(f"{data_hourly_dir}/{geo}-{name}.csv", parse_dates=['date'], date_parser=date_parser).set_index('date') for name in topics}
print(df_hourly[topic_used])
df_hourly_drop = df_hourly[topic_used][df_hourly[topic_used]["batch_id"] != 20]
df_hourly = df_hourly[topic_used]


def merge_trends_batches(left, right, overlap_hour, topic):
    """
    return the concatenation of left and right, correctly scaled based on their overlap (in hours)

    :param left: accumulator dataframe
    :param right: new dataframe to add to the accumulator
    :param overlap_hour: number of hours that are overlapping
    :param topic: topic considered
    """
    if left.empty:
        return right
    # retrieve the overlapping points:
    overlap_start = right.index.min()
    overlap_end = overlap_start + datetime.timedelta(hours=overlap_hour - 1)
    left_overlap = left[overlap_start:overlap_end]
    right_overlap = right[overlap_start:overlap_end]
    scaling = (left_overlap[topic] / right_overlap[topic]).mean()
    if scaling < 1:  # right has not the good scale
        right_to_add = right[right.index > overlap_end]
        right_to_add = right_to_add * scaling
        return left.append(right_to_add)
    else:  # left has not the good scale
        left_to_add = left[left.index < overlap_start]
        left_to_add = left_to_add / scaling
        return left_to_add.append(right)


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

        batch_df = df[df["batch_id"] == j]
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

    # drop the period at the beginning and the end, in order to begin from YYYY-MM-DD:0h ->
    for i in range(len(list_scaled_df)):
        df = list_scaled_df[i]
        old_begin, old_end = df.index.min(), df.index.max()
        new_begin = old_begin + datetime.timedelta(hours=((24 - old_begin.hour) % 24))
        new_end = old_end - datetime.timedelta(hours=((old_end.hour + 1) % 24))
        list_scaled_df[i] = df[new_begin:new_end]
    return list_scaled_df


scaled_df_drop = scale_df(df_hourly_drop, topic_code)[1:3]
scaled_df = scale_df(df_hourly, topic_code)[1]


# collect 100 daily req. and average them
def mean_query(number, begin, end, topic, geo, cat=0):
    """
    provide multiple queries on the period begin->end. the column topic contains the mean of the queries
    the queries use different interval in order to provide different results
    """

    def dates_to_timeframe(a, b):
        return f"{a.strftime('%Y-%m-%d')} {b.strftime('%Y-%m-%d')}"

    def timeframe_iterator():
        """
        yield different timeframes such that [begin, end] is always in the timeframe provided
        uses the closest dates possible to provide the right number of timeframe
        """
        # maximum date allowed for google trends
        max_end = datetime.datetime.today() - datetime.timedelta(days=4)
        lag = int(np.ceil(np.sqrt(number)))
        max_end_lag = (max_end - end).days + 1

        # compute possible end and corresponding beginning
        if lag > max_end_lag:
            lag_end = max_end_lag
            lag_begin = int(np.ceil(number / lag_end))
        else:
            lag_begin = lag
            lag_end = lag

        # yield timeframes
        for i in range(lag_begin):
            for j in range(lag_end):
                begin_tf = begin - datetime.timedelta(days=i)
                end_tf = end + datetime.timedelta(days=j)
                yield dates_to_timeframe(begin_tf, end_tf)

    df_tot = pd.DataFrame()
    cnt = 0
    pytrends = TrendReq(retries=2, backoff_factor=0.1)
    for k, timeframe in enumerate(timeframe_iterator()):
        done = False
        print(f"timeframe= {timeframe} ({k + 1}/{number})")
        while not done:
            try:
                pytrends.build_payload([topic], timeframe=timeframe, geo=geo, cat=cat)
                df = pytrends.interest_over_time()
                done = True
            except:
                sleep(10 + 10 * np.random.random())
        df = df[begin:end]
        sleep(1 + np.random.random())
        if 100 not in df[topic]:
            df[topic] = df[topic] * 100 / df[topic].max()
        df_tot[f"{topic}_{cnt}"] = df[topic]
        cnt += 1
        if cnt >= number:
            df_tot[topic_code] = df_tot.mean(axis=1)
            df_tot[topic_code] = 100 * df_tot[topic_code] / df_tot[topic_code].max()
            return df_tot


begin = datetime.datetime.strptime("2020-04-25", "%Y-%m-%d")
end = datetime.datetime.strptime("2020-05-26", "%Y-%m-%d")
df_interval = mean_query(100, begin, end, topic_code, geo)


def error_no_impute(df_true, df_recompose):
    index_interval_a = df_true.index.intersection(df_recompose[0].index)
    index_interval_b = df_true.index.intersection(df_recompose[1].index)
    MAE = max(np.mean(abs(df_true.iloc[index_interval_a] - df_recompose[0].iloc[index_interval_a])),
              np.mean(abs(df_true.iloc[index_interval_b] - df_recompose[1].iloc[index_interval_b])))
    MSE = max(np.mean((df_true.iloc[index_interval_a] - df_recompose[0].iloc[index_interval_a])**2),
              np.mean((df_true.iloc[index_interval_b] - df_recompose[1].iloc[index_interval_b]) ** 2))
    return MAE, MSE


def error_impute(df_true, df_impute, interval):
    MAE = np.mean(abs(df_true.iloc[interval] - df_impute.iloc[interval]))
    MSE = np.mean((df_true.iloc[interval] - df_impute.iloc[interval])**2)
    return MAE, MSE


def rescale_batch(df_left, df_right, df_daily):
    pass


rolling_average_before = list(range(1, 16, 2))  # hourly
rolling_average_after = list(range(1, 10, 2))  # daily agg.
rolling_daily = list(range(1, 10, 2))  # daily

for rolling_before in rolling_average_before:
    for rolling_after in rolling_average_after:
        for rolling_day in rolling_daily:
            # preprocess batch left
            batch_left = scaled_df_drop[0].rolling(rolling_before, center=True).mean().dropna()  # hourly
            batch_left = batch_left.resample('D').mean()
            batch_left = batch_left.rolling(rolling_after, center=True).mean().dropna()  # daily
            batch_left = batch_left * 100 / max(batch_left)

            # preprocess batch right
            batch_right = scaled_df_drop[1].rolling(rolling_before, center=True).mean().dropna()  # hourly
            batch_right = batch_right.resample('D').mean()
            batch_right = batch_right.rolling(rolling_after, center=True).mean().dropna()  # daily
            batch_right = batch_right * 100 / max(batch_right)
            # preprocess daily
            daily = df_interval.rolling(rolling_day, center=True).mean().dropna()  # daily
            batch_rescaled = rescale_batch(batch_left, batch_right, daily)

