from pytrends.request import TrendReq
from pytrends import exceptions
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError
import pandas as pd
import os.path
from datetime import date, datetime, timedelta
import numpy as np
from typing import List, Iterator, Tuple, Dict, Union, Type
from copy import deepcopy
import random
from collections import deque
import re
import time
from functools import partial, reduce
import requests
import json
from requests.packages.urllib3.util.retry import Retry
import util

try:
    from toripchanger import TorIpChanger
    from toripchanger.exceptions import TorIpError

    TorIpChanger(tor_password='my password', tor_port=9051,
                 local_http_proxy='127.0.0.1:8118', new_ip_max_attempts=30, reuse_threshold=0)
    tor_ip_set = True
except (ImportError, TorIpError):
    tor_ip_set = False

dir_daily = "../data/trends/collect_daily"
dir_daily_gap = "../data/trends/collect_gap"
dir_hourly = "../data/trends/collect"
dir_min_hourly = "../data/trends/collect_minimal/hourly"
dir_min_daily = "../data/trends/collect_minimal/daily"
dir_model = "../data/trends/model"
dir_explore = "../data/trends/explore"
day_format = "%Y-%m-%d"
hour_format = "%Y-%m-%dT%H"
max_query_days = 269
google_trends_errors = (exceptions.ResponseError, ReadTimeout, ConnectTimeout, TorIpError)


def date_to_datetime(x):
    return datetime(x.year, x.month, x.day) if isinstance(x, date) else x


def date_parser_daily(x):
    return datetime.strptime(x, '%Y-%m-%d')


def date_parser_hourly(x) -> datetime:
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def timeframe_to_date(timeframe: str) -> Tuple[datetime, datetime]:
    """
    transform a timeframe in daily / hourly format to a tuple of datetime
    """
    begin, end = timeframe.split()
    if is_timeframe_hourly(timeframe):  # assume that the format is a hourly format
        date_format = hour_format  # hourly formatting
    else:
        date_format = day_format  # daily formatting
    return datetime.strptime(begin, date_format), datetime.strptime(end, date_format)


def dates_to_timeframe(begin: Union[datetime, date], end: Union[datetime, date], hours=False) -> str:
    """
    transform two dates into a timeframe
    """
    if hours:
        return f"{begin.strftime(hour_format)} {end.strftime(hour_format)}"
    else:
        return f"{begin.strftime(day_format)} {end.strftime(day_format)}"


def is_timeframe_hourly(timeframe: str) -> bool:  # true if a timeframe is in hourly format
    return 'T' in timeframe


def time_delta_day(x):
    return timedelta(days=x)


def time_delta_hour(x):
    return timedelta(hours=x)


class TorTrendReq(TrendReq):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_data(self, url, method=TrendReq.GET_METHOD, trim_chars=0, **kwargs):
        """
        Shameless copy of the default method of pytrends.TrendReq, using a tor session instead of a regular one
        Send a request to Google and return the JSON response as a Python object
        :param url: the url to which the request will be sent
        :param method: the HTTP method ('get' or 'post')
        :param trim_chars: how many characters should be trimmed off the beginning of the content of the response
            before this is passed to the JSON parser
        :param kwargs: any extra key arguments passed to the request builder (usually query parameters or data)
        """
        s = requests.session()
        s.proxies = {
            'http': 'socks5h://localhost:9050',  # default port for tor
            'https': 'socks5h://localhost:9050',
        }
        # Retries mechanism. Activated when one of statements >0 (best used for proxy)
        if self.retries > 0 or self.backoff_factor > 0:
            retry = Retry(total=self.retries, read=self.retries,
                          connect=self.retries,
                          backoff_factor=self.backoff_factor)

        s.headers.update({'accept-language': self.hl})
        if len(self.proxies) > 0:
            self.cookies = self.GetGoogleCookie()
            s.proxies.update({'https': self.proxies[self.proxy_index]})
        if method == TrendReq.POST_METHOD:
            response = s.post(url, timeout=self.timeout,
                              cookies=self.cookies, **kwargs,
                              **self.requests_args)  # DO NOT USE retries or backoff_factor here
        else:
            response = s.get(url, timeout=self.timeout, cookies=self.cookies,
                             **kwargs, **self.requests_args)  # DO NOT USE retries or backoff_factor here
        # check if the response contains json and throw an exception otherwise
        # Google mostly sends 'application/json' in the Content-Type header,
        # but occasionally it sends 'application/javascript
        # and sometimes even 'text/javascript
        if response.status_code == 200 and 'application/json' in \
                response.headers['Content-Type'] or \
                'application/javascript' in response.headers['Content-Type'] or \
                'text/javascript' in response.headers['Content-Type']:
            # trim initial characters
            # some responses start with garbage characters, like ")]}',"
            # these have to be cleaned before being passed to the json parser
            content = response.text[trim_chars:]
            # parse json
            self.GetNewProxy()
            return json.loads(content)
        else:
            # error
            raise exceptions.ResponseError(
                'The request failed: Google returned a '
                'response with code {0}.'.format(response.status_code),
                response=response)

    def GetGoogleCookie(self):
        """
        Gets google cookie (used for each and every proxy; once on init otherwise)
        Removes proxy from the list on proxy error
        """
        proxies = {
            'http': 'socks5h://localhost:9050',  # default port for tor
            'https': 'socks5h://localhost:9050',
        }
        return dict(filter(lambda i: i[0] == 'NID', requests.get(
            'https://trends.google.com/?geo={geo}'.format(
                geo=self.hl[-2:]),
            timeout=self.timeout,
            proxies=proxies,
            **self.requests_args
        ).cookies.items()))


class TrendsRequest:  # interface to query google trends data, using a TrendReq instance

    def __init__(self, *args, **kwargs):
        self.request_done = 0
        self.nb_exception = 0

    def get_interest_over_time(self, kw_list, cat=0, timeframe='today 5-y', geo='', gprop='') -> Union[
        None, pd.DataFrame]:
        raise NotImplementedError

    def related_topics(self, kw_list, cat=0, timeframe='today 5-y', geo='', gprop='') -> Union[None, Dict[str, str]]:
        raise NotImplementedError

    @classmethod
    def zero_dataframe(cls, kw_list: List[str], begin: datetime, end: datetime, freq='D') -> pd.DataFrame:
        """
        return a dataframe filled with zeros for all keywords in the list, with a isPartial columns and indexed
        by datetime
        :param kw_list: columns of the resulting dataframe
        :param begin: first date for the dataframe
        :param end: last date for the dataframe
        :param freq: time difference used between two index values
        :return: dataframe filled with 0 for all keywords, with a isPartial columns and time indexed
        """
        index = pd.date_range(start=begin, end=end, freq=freq, name='date')
        data = {**{'isPartial': [False for _ in range(len(index))]},
                **{kw: np.zeros(len(index)) for kw in kw_list}}
        return pd.DataFrame(data=data, index=index)

    def __str__(self):
        return f'{self.request_done} queries done, {self.nb_exception} exceptions caught'


class LocalTrendsRequest(TrendsRequest):
    """
    process trends using local IP address. When an error is encountered, do a time.sleep before continuing
    """

    def __init__(self, max_errors, verbose=True, *args, **kwargs):
        """
        :param max_errors: max number of errors that can be processed. When max_errors errors have been caught, raise an
            error instead of sleeping the program
        """
        super().__init__(*args, **kwargs)
        self.pytrends = self.setup_pytrends()
        self.kw_list = None
        self.timeframe = None
        self.geo = None
        self.max_error = max_errors
        self.verbose = verbose
        self.error_interval = {}

    def setup_pytrends(self):
        while True:
            try:
                pytrends = TrendReq()
                return pytrends
            except ConnectionError as err:
                self.nb_exception += 1
                if self.nb_exception >= self.max_error:  # too many errors have been caught, stop the collect of data
                    raise err

    def get_interest_over_time(self, kw_list: List[str], cat: int = 0, timeframe: str = 'today 5-y', geo: str = '',
                               gprop: str = '') -> Union[pd.DataFrame, None]:
        """
        get the interest over time value for a list of keywords, on a given loc and time interval. In case of error,
        sleep and return None
        :param kw_list: keywords / topics to query
        :param cat: google trends category to use
        :param timeframe: dates to query
        :param geo: localisation of query
        :param gprop: google trends property to filter on
        :return: pandas dataframe is the request succeeded and None otherwise
        """
        self.kw_list = kw_list
        self.timeframe = timeframe
        self.geo = geo
        error_before = self.nb_exception
        if self.verbose:
            print(f'query on {self.geo}: {self.kw_list} for {timeframe}... ', end='')
        if self.timeframe not in self.error_interval:
            self.error_interval[self.timeframe] = 0
        # if self.error_interval[self.timeframe] > 2:
        if False:
            if self.verbose:
                print('ignoring time interval, ', end='')
            df = pd.DataFrame()
        else:
            function = partial(self.pytrends.build_payload, kw_list, cat, timeframe, geo, gprop)
            res = self._handle_errors(function)
            if self.nb_exception == error_before:
                self.intermediate_sleep()
            else:
                if self.verbose:
                    print('failed')
                return None
            function = partial(self.pytrends.interest_over_time)
            df = self._handle_errors(function)
            if df is None:
                if self.verbose:
                    print('failed')
                return None
            self.request_done += 1
        if df.empty:
            begin, end = timeframe_to_date(self.timeframe)
            self.error_interval[self.timeframe] += 1
            freq = 'H' if is_timeframe_hourly(self.timeframe) else 'D'
            if self.verbose:
                print('got empty dataframe')
            return TrendsRequest.zero_dataframe(self.kw_list, begin, end, freq=freq)
        else:
            self.error_interval[self.timeframe] -= 1
        if self.verbose:
            print('retrieved')
        return df

    def related_topics(self, kw_list, cat=0, timeframe='today 3-m', geo='', gprop=''):
        """
        get the interest over time value for a list of keywords, on a given loc and time interval. In case of error,
        sleep and return None
        :param kw_list: list of keywords. Should contain only one keyword
        :param cat: google trends category to use
        :param timeframe: dates to query
        :param geo: localisation of query
        :param gprop: google trends property to filter on
        :return: Dict of topics if the request succeeded and None otherwise
        """
        assert len(kw_list) == 1
        self.kw_list = kw_list
        self.timeframe = timeframe
        self.geo = geo
        error_before = self.nb_exception
        if self.verbose:
            print(f'related query on {self.geo}: {self.kw_list} for {timeframe}... ', end='')
        function = partial(self.pytrends.build_payload, kw_list, cat, timeframe, geo, gprop)
        res = self._handle_errors(function)
        if self.nb_exception == error_before:
            self.intermediate_sleep()
        else:
            if self.verbose:
                print('failed')
            return None
        function = partial(self.pytrends.related_topics)
        df = self._handle_errors(function)
        if df is None or df.empty:
            if self.verbose:
                print('failed')
            return None
        self.request_done += 1
        if self.verbose:
            print('retrieved')
        if df[kw_list[0]]['rising'].empty:
            return {}
        else:
            dic_rising = df[kw_list[0]]['rising'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()[
                'topic_mid']
            dic_top = df[kw_list[0]]['top'][['topic_mid', 'topic_title']].set_index('topic_title').to_dict()[
                'topic_mid']
            values = {**dic_rising, **dic_top}
            values = {k.replace('/', '_'): v for k, v in values.items()}
            return values

    def _handle_errors(self, function: callable):
        """
        call a function and sleep in case of error
        :param function: function to call
        :return: result of the function
        """
        fetched = False
        while not fetched:
            try:
                result = function()
                return result
            except Exception as err:
                print('exception')
                if isinstance(err, exceptions.ResponseError) and err.response.status_code == 500:
                    return pd.DataFrame()
                self.nb_exception += 1
                if self.nb_exception >= self.max_error:  # too many errors have been caught, stop the collect of data
                    raise err
                print('sleeping... ', end='')
                self.error_sleep()
                print('resuming. ', end='')
                return None

    def intermediate_sleep(self):  # sleep between 2 requests
        time.sleep(random.uniform(1, 2))

    def error_sleep(self):
        if self.nb_exception < 3:
            time.sleep(60 + random.uniform(30, 90))
        else:
            time.sleep(60 * random.randint(1, self.nb_exception) + random.uniform(30, 90))


class TorTrendsRequest(LocalTrendsRequest):

    def __init__(self, *args, **kwargs):
        self.tor_ip_changer = None
        self.reset(True)
        super().__init__(*args, **kwargs)
        self.error_since_ip_change = 0  # errors since the last change of ip
        self.error_change_ip = 3  # number of errors that must be reached in order to change ip (with a certain prob.)

    def reset(self, first=False):  # set a new TorTrendReq instance and a new tor_ip_changer (with a new ip address)
        self.tor_ip_changer = TorIpChanger(tor_password='my password', tor_port=9051,
                                           local_http_proxy='127.0.0.1:8118', new_ip_max_attempts=30, reuse_threshold=0)
        if not first:
            self.get_new_ip()

    def setup_pytrends(self):
        return self.get_new_ip()

    def get_new_ip(self):  # get a new ip address for tor and print it
        running = True
        pytrends = None
        while running:
            try:
                self.tor_ip_changer.get_new_ip()
                self.error_since_ip_change = 0
                self.intermediate_sleep()
                print(f'tor ip for next requests = {self.current_tor_ip()}. ', end='')
                self.intermediate_sleep()
                pytrends = TorTrendReq()
                running = False
            except (exceptions.ResponseError, ReadTimeout, ConnectTimeout) as err:
                pass
        return pytrends

    def _handle_errors(self, function: callable):
        """
        call a function and sleep / change ip in case of error.
        :param function: function to call
        :return: result of the function
        """
        fetched = False
        count_error = 0
        while not fetched:
            try:
                try:
                    result = function()
                    return result
                except (exceptions.ResponseError, ReadTimeout, ConnectTimeout) as err:
                    # timeframe not available
                    if isinstance(err, exceptions.ResponseError) and err.response.status_code == 500:
                        return pd.DataFrame()
                    count_error += 1
                    self.nb_exception += 1
                    self.error_since_ip_change += 1
                    if count_error >= self.max_error:
                        raise err
                    print('sleeping...', end='')
                    self.error_sleep()
                    print('resuming ', end='')
                    # certain probability of changing ip. The ip is not changed at every error caught in order to fool
                    # google trends
                    if random.randint(self.error_since_ip_change,
                                      self.error_since_ip_change + 2) >= self.error_change_ip:
                        self.get_new_ip()
                    return None
            except (Exception) as err:  # error with tor, set a new tor ip changer
                count_error += 1
                self.nb_exception += 1
                if count_error >= self.max_error:
                    raise err
                self.error_sleep()
                self.reset()
                return None

    @classmethod
    def current_tor_ip(cls) -> str:  # return the current ip used by tor
        s = requests.session()
        s.proxies = {
            'http': 'socks5h://localhost:9050',  # default port for tor
            'https': 'socks5h://localhost:9050',
        }
        r = s.get('http://httpbin.org/ip')
        return f'''{r.text.split('"')[3]}'''

    def __str__(self):
        return super().__str__() + f'. Current tor ip = {self.current_tor_ip()}'

    '''
    def error_sleep(self):  # sleep done when an error is encountered
        if self.nb_exception < 3:
            time.sleep(60 + random.uniform(30, 90))
        else:
            time.sleep(60 * random.randint(1, min(self.nb_exception, 5)) + random.uniform(30, 90))
    '''

    def intermediate_sleep(self):  # sleep between 2 requests
        time.sleep(random.uniform(1, 2))

    def error_sleep(self):  # sleep between 2 requests
        if random.random() < 0.95:
            self.intermediate_sleep()
        else:
            time.sleep(random.uniform(30, 45))


class FakeTrendsRequest(TrendsRequest):  # for testing purposes, use fake trends

    def __init__(self, errors: List[datetime] = None, verbose: bool = True, *args, **kwargs):
        """
        :param errors: list of dates for which the query should return a dataframe filled with zeros
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.kw_list = []
        self.geo = ''
        self.begin = None
        self.end = None
        self.errors = [] if errors is None else errors  # list of dates that should throw an error when queried

    def build_payload(self, kw_list: List[str], cat: int = 0, timeframe: str = 'today 5-y', geo: str = '',
                      gprop: str = ''):
        self.begin, self.end = timeframe_to_date(timeframe)
        self.geo = geo
        self.kw_list = kw_list

    def get_interest_over_time(self, kw_list, cat=0, timeframe='today 5-y', geo='', gprop=''):
        self.begin, self.end = timeframe_to_date(timeframe)
        self.geo = geo
        self.kw_list = kw_list
        if self.verbose:
            print(f'query on {self.geo}: {self.kw_list} for {timeframe}')
        if self.begin is None:
            return KeyError  # pytrends behavior
        delta = self.end - self.begin
        if delta.days > max_query_days:
            freq = timedelta(days=7)
        elif delta.days > 7:
            freq = 'D'
        else:
            freq = 'H'
        for date in self.errors:  # check if the request should be valid or not
            if self.begin <= date <= self.end:
                return self.zero_dataframe(self.kw_list, self.begin, self.end, freq=freq)
        index = pd.date_range(start=self.begin, end=self.end, freq=freq, name='date')
        data = {'isPartial': [False for _ in range(len(index))]}
        for kw in self.kw_list:
            values = np.random.randint(0, 99, len(index))
            index_100 = np.random.randint(len(index))
            values[index_100] = 100
            data[kw] = values
        df = pd.DataFrame(data={**data, 'date': index}).set_index('date')
        return df

    def interest_over_time(self):
        if self.begin is None:
            return KeyError  # pytrends behavior
        delta = self.end - self.begin
        if delta.days > max_query_days:
            freq = timedelta(days=7)
        elif delta.days > 7:
            freq = 'D'
        else:
            freq = 'H'
        for date in self.errors:  # check if the request should be valid or not
            if self.begin <= date <= self.end:
                return self.zero_dataframe(self.kw_list, self.begin, self.end, freq=freq)
        index = pd.date_range(start=self.begin, end=self.end, freq=freq, name='date')
        data = {'isPartial': [False for _ in range(len(index))]}
        for kw in self.kw_list:
            values = np.random.randint(0, 99, len(index))
            index_100 = np.random.randint(len(index))
            values[index_100] = 100
            data[kw] = values
        df = pd.DataFrame(data=data, index=index)
        return df


class QueryBatch:
    """
    query done on a single batch of data. This query is assigned a batch id
    """

    def __init__(self, kw: str, geo: str, trends_request: TrendsRequest,
                 begin: Union[datetime, date], end: Union[datetime, date],
                 batch_id: int, cat: int = 0, number: int = 1, gprop: str = '', shuffle: bool = True,
                 *args, **kwargs):
        """
        process a google trends query on a single batch
        :param kw: keyword to process
        :param geo: geo code to query on
        :param trends_request: TrendsRequest instance to use to process the queries
        :param begin: first date to query in the batch
        :param end: last date to query in the batch
        :param batch_id: id of the batch
        :param cat: google trends category to use
        :param number: number of queries done on a single batch
        :param gprop: google trends property to filter on
        :param shuffle: whether to shuffle the list of queries done on the batch or not
        """
        self.kw = kw
        self.geo = geo
        self.trends_request = trends_request
        self.begin = begin
        self.end = end
        self.batch_id = batch_id
        self.cat = cat
        self.number = number
        self.gprop = gprop
        self.shuffle = shuffle
        self.df = pd.DataFrame()
        self.timeframe = dates_to_timeframe(begin, end)
        self.delta = end - begin

    @classmethod
    def latest_day_available(cls):  # latest day available for this type of batch query
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> bool:
        """
        process the batch query and return True if it is completely processed
        :return: True if the batch query is processed
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError

    def remaining(self) -> int:
        raise NotImplementedError

    @classmethod
    def max_len(cls) -> int:  # max number of data points that can be queried with one request of this type
        raise NotImplementedError

    def fetch_data(self, timeframe: str) -> pd.DataFrame:  # fetch google trends data
        df = self.trends_request.get_interest_over_time([self.kw], cat=self.cat, timeframe=timeframe, geo=self.geo,
                                                        gprop=self.gprop)
        return df

    def get_df(self) -> pd.DataFrame:  # get the dataframe for this query batch, with a batch_id column
        df = self.df
        df['batch_id'] = self.batch_id
        return df

    def get_batch_id(self) -> int:
        return self.batch_id

    def set_trends_request(self, trends_request) -> None:  # change the trends trends_request used
        self.trends_request = trends_request


class DailyQueryBatch(QueryBatch):
    """
    Query batch whose final result is based on the mean of several queries
    """

    def __init__(self, kw: str, geo: str, trends_request: TrendsRequest, begin: Union[datetime, date],
                 end: Union[datetime, date], batch_id: int, cat: int = 0, number: int = 1, gprop: str = '',
                 shuffle: bool = True):
        """
        process a daily google trends query on a single batch
        :param kw: keyword to process
        :param geo: geo code to query on
        :param trends_request: TrendsRequest instance to use to process the queries
        :param begin: first date to query in the batch
        :param end: last date to query in the batch
        :param batch_id: id of the batch
        :param cat: google trends category to use
        :param number: number of queries done on a single batch. Final result = mean of those queries
        :param gprop: google trends property to filter on
        :param shuffle: whether to shuffle the list of queries done on the batch or not
        """
        super().__init__(kw, geo, trends_request, begin, end, batch_id, cat, number, gprop, shuffle=shuffle)
        self.list_dates = list(self.dates_iterator(self.begin, self.end, self.number))
        if self.shuffle:
            random.shuffle(self.list_dates)
        self.date_idx = 0

    @classmethod
    def latest_day_available(cls):  # latest day that can be safely queried by google trends
        latest = date.today() - timedelta(days=3)
        return datetime(latest.year, latest.month, latest.day)

    @classmethod
    def max_len(cls) -> int:
        return max_query_days + 1

    def __call__(self, *args, **kwargs) -> bool:
        """
        select one interval of dates and fetch its data
        :return:
        """
        if self.date_idx < self.number:
            begin, end = self.list_dates[self.date_idx]
            timeframe = dates_to_timeframe(begin, end, hours=False)
            df = self.fetch_data(timeframe)
            if df is None:  # the request could not be fetched
                return False
            df = df[self.begin:self.end]
            if 100 not in df[self.kw]:
                self.df[f'{self.kw}_{self.date_idx}'] = df[self.kw] * 100 / df[self.kw].max()
            else:
                self.df[f'{self.kw}_{self.date_idx}'] = df[self.kw]
            self.date_idx += 1
            if self.date_idx == self.number:  # no more dates to query
                self.df[self.kw] = self.df.mean(axis=1)
                self.df[self.kw] = 100 * self.df[self.kw] / self.df[self.kw].max()
                return True
            return False
        return True

    def is_finished(self) -> bool:
        return self.date_idx == self.number

    def remaining(self) -> int:
        return self.number - self.date_idx

    @classmethod
    def largest_dates_iterator(cls, begin: datetime, end: datetime, number: int) -> Tuple[datetime, datetime]:
        """
        gives the largest interval of dates that will be used by dates_iterator
        :param begin: date of beginning for the query
        :param end: date of end for the query
        :param number: amount of queries that must cover this interval
        :return largest interval of dates that will be used
        """
        min_date = datetime.today()
        max_date = datetime.strptime("1990-01-01", "%Y-%m-%d")
        for date_a, date_b in cls.dates_iterator(begin, end, number):
            min_date = min(min_date, date_a)
            max_date = max(max_date, date_b)
        return min_date, max_date

    @classmethod
    def dates_iterator(cls, begin: datetime, end: datetime, number: int) -> Iterator[Tuple[datetime, datetime]]:
        """
        return the largest interval of dates that must be available to provide number queries on the interval
        :param begin: date of beginning for the query
        :param end: date of end for the query
        :param number: amount of queries that must cover this interval
        :return yield number tuples of dates (a, b), that cover [begin, end]
            the order looks like [(a, b), (a-1, b), (a, b+1), (a-2, b), (a-2, b+1), ... ]
            a check is done to be sure that the last date can be queried. No check is done to be sure that the interval
            is not greater than max_len
        """
        max_end = cls.latest_day_available()  # maximum possible len
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
                j = lag_left - 1
                while number_given < number and j >= 0:
                    number_given += 1
                    yield begin - timedelta(days=j), wanted_end
                    j -= 1

            lag_left += 1
            lag_right += 1


class DailyGapQuery(DailyQueryBatch):  # daily query batch meant to be used on a single gap

    def __init__(self, topic_name: str, topic_code: str, geo: str, trends_request: TrendsRequest, directory: str,
                 begin: Union[datetime, date], end: Union[datetime, date],
                 batch_id: int, cat: int = 0, number: int = 1, gprop: str = '', shuffle: bool = True,
                 savefile: bool = True):
        """
        :param topic_name: name of the topic
        :param topic_code: code of the topic
        :param geo: geo code to query on
        :param trends_request: TrendsRequest instance to use to process the queries
        :param begin: first date to query in the batch
        :param end: last date to query in the batch
        :param batch_id: id of the batch
        :param cat: google trends category to use
        :param number: number of queries done on a single batch. Final result = mean of those queries
        :param gprop: google trends property to filter on
        :param shuffle: whether to shuffle the list of queries done on the batch or not
        :param savefile: whether to save the file or not
        """
        super().__init__(topic_code, geo, trends_request, begin, end, batch_id, cat, number, gprop, shuffle=shuffle)
        self.directory = directory
        self.topic_name = topic_name
        self.savefile = savefile

    def __str__(self):
        return f'{self.directory}/{self.geo}-{self.topic_name}-{self.timeframe.replace(" ", "-")}'

    def get_df(self) -> pd.DataFrame:  # no batch id is added here
        return self.df

    def __call__(self, *args, **kwargs) -> bool:
        finished = super().__call__(*args, **kwargs)
        if finished and self.savefile:
            self.to_csv()
        return finished

    def to_csv(self):
        self.df.to_csv(f'{str(self)}.csv')


class HourlyQueryBatch(QueryBatch):  # query batch using a hourly interval

    def __init__(self, kw: str, geo: str, trends_request: TrendsRequest, begin: Union[datetime, date],
                 end: Union[datetime, date], batch_id: int, cat: int = 0, number: int = 1, gprop: str = '', *args,
                 **kwargs):
        """
        process a daily google trends query on a single batch
        :param kw: keyword to process
        :param geo: geo code to query on
        :param trends_request: TrendsRequest instance to use to process the queries
        :param begin: first date to query in the batch
        :param end: last date to query in the batch
        :param batch_id: id of the batch
        :param cat: google trends category to use
        :param number: number of queries done on a single batch. Final result = mean of those queries
        :param gprop: google trends property to filter on
        """
        self.asked_begin = begin
        self.asked_end = end
        begin, end = self.ensure_min_hour(begin, end)
        super().__init__(kw, geo, trends_request, begin, end, batch_id, cat, number, gprop)
        self.timeframe = dates_to_timeframe(self.begin, self.end, hours=True)

    @classmethod
    def latest_day_available(cls):
        latest = datetime.now().replace(microsecond=0, second=0, minute=0) - timedelta(hours=1)
        if latest.hour != 23:
            latest = latest.replace(hour=23) - timedelta(days=1)
        return latest

    @classmethod
    def max_len(cls) -> int:
        return 192  # number of data points allowed by google trends when using hourly requests

    @classmethod
    def ensure_min_hour(cls, begin: datetime, end: datetime) -> Tuple[datetime, datetime]:
        """
        ensure that the duration is the minimum allowed to get hourly requests
        """
        if end - begin < timedelta(days=3):
            return end - timedelta(days=3), end
        else:
            return begin, end

    def __call__(self, *args, **kwargs) -> bool:
        if self.df.empty:
            df = self.fetch_data(self.timeframe)
            if df is None:  # the request could not be fetched
                return False
            self.df = df.drop(columns=['isPartial'])[self.asked_begin:self.asked_end]
            if sum(self.df.index.minute):
                # if the dataframe was not given as hourly (for instance minute dataframe), resample it to daily data
                self.df = self.df.resample('H').mean()
                print(f'resampling data on {self.timeframe} to hourly data')
            # check if the dataframe is valid or not. if not, a negative batch id is set
            # a valid batch must have at least more than 3 values above 10
            if len(np.where(self.df[self.kw] > 10)[0]) <= 3:
                self.batch_id = -self.batch_id
        return True

    def is_finished(self) -> bool:
        return not self.df.empty

    def remaining(self) -> int:
        return 1 - self.df.empty


class Query:
    def __init__(self, topic_name: str, topic_code: str, geo: str, trends_request: TrendsRequest,
                 begin: Union[datetime, date], end: Union[datetime, date],
                 directory: str, overlap: int, cat: int = 0, number: int = 1, gprop='',
                 query_batch: Type[QueryBatch] = DailyQueryBatch,
                 savefile: bool = True, freq='D', shuffle: bool = True):
        """
        a google trends query done on an interval. The query will be split into several batches, in order to cover it
        :param topic_name: name of the topic
        :param topic_code: code of the topic
        :param geo: geo code to query on
        :param trends_request: TrendsRequest instance to use to process the queries
        :param begin: first date to query in the batch
        :param end: last date to query in the batch
        :param directory: directory where to save the resulting dataframe
        :param overlap: overlap between 2 batches of query
        :param cat: google trends category to use
        :param number: number of queries done on a single batch. Final result = mean of those queries
        :param gprop: google trends property to filter on
        :param savefile: whether to save the file generated or not
        :param freq: frequency of the data: 'D' for daily or 'H' for hourly
        :param shuffle: whether to select a query batch at random or not
        """
        assert type(begin) == type(end)
        self.topic_name = topic_name
        self.topic_code = topic_code
        self.geo = geo
        self.trends_request = trends_request
        self.begin = begin
        self.end = min(end, query_batch.latest_day_available())
        self.directory = directory
        self.overlap = overlap
        self.cat = cat
        self.number = number
        self.gprop = gprop
        self.query_batch = query_batch
        self.savefile = savefile
        self.freq = freq
        self.shuffle = shuffle
        self.delta = end - begin
        self.df = pd.DataFrame()
        self.list_new_requests = []
        if freq == 'D':
            self.delta_overlap = time_delta_day(self.overlap - 1)
            self.date_parser = date_parser_daily
            self.time_delta = time_delta_day
        elif freq == 'H':
            self.delta_overlap = time_delta_hour(self.overlap - 1)
            self.date_parser = date_parser_hourly
            self.time_delta = time_delta_hour
        else:
            raise Exception(f'frequency not supported: {freq}')
        self.prepare_query()

    def __call__(self, *args, **kwargs) -> bool:
        if not self.list_new_requests:
            return True
        idx = random.randrange(len(self.list_new_requests)) if self.shuffle else 0
        request_batch = self.list_new_requests[idx]
        finished = request_batch()
        if finished:
            df = request_batch.get_df()
            if self.df.empty:
                self.df = df
            else:
                batch_id = abs(request_batch.get_batch_id())
                df_a = self.df[abs(self.df['batch_id']) < batch_id]
                df_b = self.df[abs(self.df['batch_id']) > batch_id]
                self.df = pd.concat([df_a, df, df_b])
            if self.savefile:  # save to csv
                self.to_csv()
            if self.shuffle:
                self.list_new_requests.pop(idx)
            else:
                self.list_new_requests = self.list_new_requests[1:]
            if not self.list_new_requests:  # no more requests must be done
                return True
        return False

    def is_finished(self):
        return len(self.list_new_requests) == 0

    def remaining(self):
        return len(self.list_new_requests)

    def dates_interval_batches(self, begin, end) -> List[Tuple[datetime, datetime]]:
        """
        return the dates used to query the interval between begin and end. Each interval has an overlap of self.overlap
        elements with its neighbor
        :param begin: beginning of the interval to query
        :param end: end of the interval to query
        """
        if type(begin) is date:
            cur_begin = date_to_datetime(begin)
        else:
            cur_begin = deepcopy(begin)
        if type(end) is date:
            end = date_to_datetime(end)

        latest_day = min(self.query_batch.latest_day_available(), end)

        # max lag used by dates iterator if every date can be queried
        max_lag_left = np.floor(np.sqrt(self.number - 1))  # max left lag
        max_lag_right = np.floor(np.sqrt(self.number - max_lag_left - 1))  # max right lag
        max_len = self.time_delta(
            self.query_batch.max_len() - 1 - max_lag_left - max_lag_right)  # max length for a query
        delta = max_len - self.time_delta(self.overlap - 1)
        delta_lag_right = self.time_delta(max_lag_right)
        # max_lag_right == 0 for the latest date, as it is impossible to ask data for a date > latest_day
        max_len_end = self.time_delta(self.query_batch.max_len() - self.number)

        list_dates = []
        cur_end = cur_begin + max_len
        while cur_end + delta_lag_right < latest_day:  # while queries can be safely done
            list_dates.append((cur_begin, cur_end))
            cur_begin += delta
            cur_end += delta
        # need to add the last pair of dates that must be queried
        if (latest_day - cur_begin) <= max_len_end:  # the last pair can be safely queried
            list_dates.append((cur_begin, latest_day))
        else:  # the last query must be split in two
            cur_end_b = latest_day
            cur_begin_b = latest_day - max_len_end
            cur_begin_a = cur_begin
            cur_end_a = cur_begin_a + max_len_end
            # cur_end_a = cur_begin_b + delta_overlap
            # cur_begin_a = cur_begin
            list_dates.append((cur_begin_a, cur_end_a))
            list_dates.append((cur_begin_b, cur_end_b))
        return list_dates

    def list_dates(self) -> List[Tuple[datetime, datetime]]:  # list of dates used on each batch
        return self.dates_interval_batches(self.begin, self.end)

    def prepare_query(self):
        """
        try to read the existing queries and see if they must be actualized or not
        set the values for
        - self.df
        - self.list_new_requests
        """
        filename = self.__str__() + '.csv'
        if os.path.exists(filename):
            df_tot = pd.read_csv(filename, parse_dates=['date'], date_parser=self.date_parser).set_index('date')
            df_tot = df_tot
            # check if every query is present
            gb = df_tot.groupby('batch_id')
            # df_covered = sorted([df for _, df in gb], key=lambda x: abs(x.iloc[0]['batch_id']))
            df_covered = sorted([df for _, df in gb], key=lambda x: x.index.min())
            # append at the beginning if needed
            batch_id = 1
            if df_covered[0].index.min() > self.begin:  # need to provide queries earlier than what has been saved
                dates_begin = self.dates_interval_batches(self.begin, df_covered[0].index.min() + self.delta_overlap)
                for begin, end in dates_begin:
                    self.list_new_requests.append(self.query_batch(self.topic_code, self.geo, self.trends_request,
                                                                   begin, end, batch_id, self.cat,
                                                                   self.number, self.gprop, shuffle=self.shuffle))
                    batch_id += 1
            # append in middle if needed
            df = df_covered[0]
            df['batch_id'] = batch_id
            # if self.freq == 'D':
            #    df = df.set_index(df.index.date, drop=True)
            valid_df = [df]
            batch_id += 1
            delta_1 = self.time_delta(1)
            for df_left, df_right in zip(df_covered, df_covered[1:]):
                intersection = len(df_left.index.intersection(df_right.index - delta_1))
                if intersection > 0 and (df_right.iloc[0]['batch_id'] < 0 or df_left.iloc[0]['batch_id'] < 0):
                    # one of the two batches was invalid and no need to query between them, as only errors would happen
                    pass
                elif intersection < self.overlap:
                    # queries should be done between the 2 dataframes
                    dates = self.dates_interval_batches(df_left.index.max() - self.delta_overlap,
                                                        df_right.index.min() + self.delta_overlap)
                    for begin, end in dates:
                        self.list_new_requests.append(self.query_batch(self.topic_code, self.geo, self.trends_request,
                                                                       begin, end, batch_id, self.cat,
                                                                       self.number, self.gprop, shuffle=self.shuffle))
                        batch_id += 1
                else:  # the batches and their intersection is valid
                    pass
                df = df_right
                # keep the sign in case the batch id was negative
                df['batch_id'] = np.sign(df.iloc[0]['batch_id']) * batch_id
                # if self.freq == 'D':
                #    df = df.set_index(df.index.date, drop=True)
                valid_df.append(df)
                batch_id += 1
            # append end if needed
            if df_covered[-1].index.max() < self.end:  # new queries must be made
                dates_end = self.dates_interval_batches(df_covered[-1].index.max() - self.delta_overlap, self.end)
                '''
                if len(df_covered) == 1:  # there was only one existing dataframe, no need to drop it
                    dates_end = self.dates_interval_batches(df_covered[-1].index.max() - self.delta_overlap, self.end)
                else:
                    valid_df.pop()  # remove the last dataframe
                    batch_id -= 1
                    dates_end = self.dates_interval_batches(valid_df[-1].index.max() - self.delta_overlap, self.end)
                '''
                for begin, end in dates_end:
                    self.list_new_requests.append(self.query_batch(self.topic_code, self.geo, self.trends_request,
                                                                   begin, end, batch_id, self.cat,
                                                                   self.number, self.gprop, shuffle=self.shuffle))
                    batch_id += 1
            self.df = pd.concat(valid_df)
        else:
            self.list_new_requests = [self.query_batch(self.topic_code, self.geo, self.trends_request,
                                                       begin, end, batch_id + 1, self.cat,
                                                       self.number, self.gprop, shuffle=self.shuffle)
                                      for batch_id, (begin, end) in enumerate(self.list_dates())]

    def __str__(self):
        return f'{self.directory}/{self.geo}-{self.topic_name}'

    def get_df(self):
        return self.df

    def to_csv(self):
        self.df.to_csv(f'{str(self)}.csv')

    def set_trends_request(self, trends_request: TrendsRequest):
        """
        set the trends request for this Query and all its query batches
        :param trends_request: TrendsRequest instance to use
        """
        self.trends_request = trends_request
        for query_batch in self.list_new_requests:
            query_batch.set_trends_request(trends_request)


class DailyQuery(Query):

    def __init__(self, topic_name: str, topic_code: str, geo: str, trends_request: TrendsRequest,
                 begin: Union[datetime, date], end: Union[datetime, date], directory: str, overlap: int,
                 cat: int = 0, number: int = 1, gprop='', savefile: bool = True, shuffle: bool = True):
        super().__init__(topic_name, topic_code, geo, trends_request, begin, end, directory, overlap, cat, number,
                         gprop, DailyQueryBatch, savefile, freq='D', shuffle=shuffle)

    def remaining(self):
        return reduce(lambda x, y: x + y.remaining(), self.list_new_requests, 0)


class HourlyQuery(Query):

    def __init__(self, topic_name: str, topic_code: str, geo: str, trends_request: TrendsRequest,
                 begin: Union[datetime, date], end: Union[datetime, date], directory: str, overlap: int,
                 cat: int = 0, number: int = 1, gprop='', savefile: bool = True, shuffle: bool = True):
        super().__init__(topic_name, topic_code, geo, trends_request, begin, end, directory, overlap, cat, number,
                         gprop, HourlyQueryBatch, savefile, freq='H', shuffle=shuffle)


class QueryList:  # handle a list of queries
    def __init__(self, topics: Dict[str, str], geo: Dict[str, str], directory: str, trends_request: TrendsRequest,
                 begin: datetime, end: datetime, query: Type[Query] = DailyQuery, query_limit: int = 0,
                 overlap: int = 30, number: int = 1, cat: int = 0, gprop: str = '', savefile: bool = True,
                 shuffle: bool = True, verbose=True):
        """
        :param topics: dict of topic_name, topic_code to query
        :param geo: dict of geo_code, geo_name to query
        :param directory: directory where to load and save the data
        :param trends_request: TrendsRequest instance used to query
        :param begin: first date to query
        :param end: last date to query
        :param query: class of query to use
        :param query_limit: max number of queries to keep in the list of queries to process. If zero, no limit is set
            on the number of queries in the list. If positive, keep up to query_limit queries into list_queries
            used to reduce memory consumption
        :param overlap: overlap between 2 batches of queries done
        :param number: number of queries done on a single batch
        :param cat: google trends category to use
        :param gprop: google trends property to filter on
        :param savefile: whether to save the results or not
        :param shuffle: whether to shuffle the requests or not
        """
        self.topics = topics
        self.geo = geo
        self.directory = directory
        self.trends_request = trends_request
        self.begin = begin
        self.end = end
        self.query = query
        self.limit = query_limit
        self.overlap = overlap
        self.number = number
        self.cat = cat
        self.gprop = gprop
        self.savefile = savefile
        self.shuffle = shuffle
        self.verbose = verbose
        self.status = 'initialized'  # one of 'initialized', 'running', 'done'
        self.list_queries = deque()
        self.in_list = 0
        self.growth_iterator = self.growth_list_iterator()
        self.max_processed = len(topics) * len(geo)
        self.nb_processed = 0  # number of query processed
        self.nb_sub_processed = 0  # number of subquery processed
        self.shuffle_cycle = 100  # every shuffle_cycle sub queries processed, shuffle the list again
        self.process_cycle = 0  # counter of sub queries processed since the last shuffle call
        self.growth_list()

    def __call__(self, *args, **kwargs) -> bool:
        """
        Choose a query at random and process it. If the query is finished, removes it from the list of queries
        :return True if all queries have been processed
        """
        if self.verbose:
            print('remaining calls:', self.remaining())
        if self.list_queries:  # there are queries awaiting
            empty_list = False
            found_finished = True
            query = None
            ignored = -1
            while found_finished and self.list_queries:  # check for queries that are already finished
                if self.shuffle:
                    # query = self.list_queries.pop()
                    query = self.list_queries[-1]
                    if query.is_finished():
                        self.list_queries.pop()
                    else:
                        found_finished = False
                else:
                    # query = self.list_queries.popleft()
                    query = self.list_queries[0]
                    if query.is_finished():
                        self.list_queries.popleft()
                    else:
                        found_finished = False
                ignored += 1
            self.nb_processed += ignored
            if not self.list_queries:
                empty_list = True
            else:
                query_finished = query()  # can throw an exception
                if self.shuffle:
                    self.list_queries.pop()
                else:
                    self.list_queries.popleft()
                self.nb_sub_processed += 1
                self.process_cycle += 1
                if query_finished:
                    self.nb_processed += 1
                    self.in_list -= 1
                    if not self.list_queries:  # no more queries in the list
                        empty_list = True
                        if self.status == 'done':
                            self.done()
                else:
                    self.list_queries.appendleft(query)
                if self.process_cycle % self.shuffle_cycle == 0 and self.shuffle:  # reshuffle the queries
                    random.shuffle(self.list_queries)
        else:  # no more queries in the list
            empty_list = True
        if empty_list:
            if self.status == 'done':  # no more query to process
                return True
            else:  # the list of queries should grow
                self.growth_list()
                return self()
        else:
            return False

    def done(self):  # called once when all queries have been processed
        pass

    def growth_list(self):
        """
        put more items into the list of queries
        """
        if self.status != 'done':
            while (self.in_list != self.limit or 0 == self.limit) and self.status != 'done':
                query = next(self.growth_iterator)
                if query is None:
                    break
                self.list_queries.append(query)
                self.in_list += 1
            if self.shuffle:
                random.shuffle(self.list_queries)
            self.process_cycle = 0

    def growth_list_iterator(self) -> Iterator[Query]:
        self.status = 'running'
        for topic_name, topic_code in self.topics.items():
            for geo_code, geo_name in self.geo.items():
                yield self.query(topic_name, topic_code, geo_code, self.trends_request, self.begin, self.end,
                                 self.directory, self.overlap, self.cat, self.number, self.gprop,
                                 savefile=self.savefile, shuffle=self.shuffle)
        self.status = 'done'
        yield None

    def set_trends_request(self, trends_request: TrendsRequest):
        """
        set the trends request for this QueryList, all its current queries and its newest queries
        :param trends_request: TrendsRequest instance to use
        """
        self.trends_request = trends_request
        for query in self.list_queries:
            query.set_trends_request(trends_request)

    def remaining(self):
        return reduce(lambda x, y: x + y.remaining(), self.list_queries, 0)


class DailyQueryList(QueryList):  # query using the daily method

    def __init__(self, topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime, query_limit: int = 0, overlap: int = 30, number: int = 2,
                 cat: int = 0, gprop: str = '', savefile: bool = True, shuffle: bool = True, directory=dir_daily):
        super().__init__(topics, geo, directory, trends_request, begin, end, DailyQuery, query_limit, overlap, number,
                         cat, gprop, savefile, shuffle)


class HourlyQueryList(QueryList):  # query using the hourly method

    def __init__(self, topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime, query_limit: int = 0, overlap: int = 15, number: int = 1,
                 cat: int = 0, gprop: str = '', savefile: bool = True, shuffle: bool = True, directory=dir_hourly):
        super().__init__(topics, geo, directory, trends_request, begin, end, HourlyQuery, query_limit, overlap, number,
                         cat, gprop, savefile, shuffle)


class DailyGapQueryList(QueryList):

    def __init__(self, topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime, query_limit: int = 0, overlap: int = 30, number: int = 10,
                 cat: int = 0, gprop: str = '', savefile: bool = True, shuffle: bool = True):
        self.directory_hourly = dir_hourly
        self.files_remove = []
        super().__init__(topics, geo, dir_daily_gap, trends_request, begin, end, DailyQuery, query_limit, overlap,
                         number, cat, gprop, savefile, shuffle)

    def growth_list_iterator(self) -> Iterator[DailyGapQuery]:
        self.status = 'running'
        for topic_name, topic_code in self.topics.items():
            for geo_code, geo_name in self.geo.items():
                filename = f'{self.directory_hourly}/{geo_code}-{topic_name}.csv'
                df_hourly = pd.read_csv(filename, parse_dates=['date'], date_parser=date_parser_hourly).set_index(
                    'date')
                list_df_hourly = HourlyModelData.hourly_to_list_daily(df_hourly, topic_code)
                starting_pattern = f"{geo_code}-{topic_name}-"  # pattern for the queries on the gap
                existing_files = [filename for filename in os.listdir(self.directory) if
                                  filename.startswith(starting_pattern)]
                list_daily_df = [pd.read_csv(f"{self.directory}/{file}", parse_dates=['date'],
                                             date_parser=date_parser_daily).set_index('date')[[topic_code]] for file in
                                 existing_files]
                dates_actualize = []
                daily_intersection = []
                if list_df_hourly[0].index.min().to_pydatetime() > self.begin:
                    # the first batches were invalid
                    found = False
                    for daily_df in list_daily_df:
                        if daily_df.index.min().to_pydatetime() <= self.begin:
                            daily_intersection.append(daily_df)
                            found = True
                            break
                    if not found:
                        # no previous daily batch is able to provide data for it
                        # if reduce(lambda x, y: x and (y.index.min().to_pydatetime() > self.begin), list_daily_df, True):
                        dates_query = self.begin, min(
                            (list_df_hourly[0].index.min() + timedelta(days=self.overlap - 1)).to_pydatetime(),
                            min(DailyQueryBatch.latest_day_available(), list_df_hourly[0].index.max().to_pydatetime()))
                        dates_actualize.append(dates_query)

                for df_left, df_right in zip(list_df_hourly, list_df_hourly[1:]):
                    df_intersection, can_be_actualized = DailyGapQueryList.find_largest_intersection(df_left, df_right,
                                                                                                     list_daily_df,
                                                                                                     overlap=self.overlap)
                    # print(df_intersection, can_be_actualized)
                    if can_be_actualized:
                        df_intersection, can_be_actualized = DailyGapQueryList.find_largest_intersection(df_left,
                                                                                                         df_right,
                                                                                                         list_daily_df,
                                                                                                         overlap=self.overlap)
                        dates_query = (df_left.index.max() - timedelta(days=self.overlap - 1)).to_pydatetime(), \
                                      min((df_right.index.min() + timedelta(days=self.overlap - 1)).to_pydatetime(),
                                          DailyQueryBatch.latest_day_available())
                        dates_query = max(df_left.index.min().to_pydatetime(), dates_query[0]), \
                                      min(df_right.index.max().to_pydatetime(), dates_query[1])
                        dates_actualize.append(dates_query)
                    else:
                        daily_intersection.append(df_intersection)
                for i, df in enumerate(list_daily_df):
                    to_remove = True
                    for df_chosen in daily_intersection:
                        if df is df_chosen:
                            to_remove = False
                            break
                    if to_remove:
                        self.files_remove.append(existing_files[i])
                min_dates = self.find_min_dates_queries(dates_actualize, self.number)
                for begin, end in min_dates:
                    yield DailyGapQuery(topic_name, topic_code, geo_code, self.trends_request, self.directory, begin,
                                        end, 0, self.cat, self.number, self.gprop, shuffle=self.shuffle,
                                        savefile=self.savefile)
        self.status = 'done'
        yield None

    def done(self):
        """
        remove the old daily gap files that were used before actualizing them
        """
        '''
        for filename in self.files_remove:
            file = f'{self.directory}/{filename}'
            if os.path.exists(file):
                os.remove(file)
        '''
        pass

    @staticmethod
    def find_largest_intersection(df_a: pd.DataFrame, df_b: pd.DataFrame, list_df_daily: List[pd.DataFrame],
                                  overlap: int = 30) -> Tuple[pd.DataFrame, bool]:
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
        best_inter_left = -1
        best_inter_right = -1
        best_df = None
        can_be_actualized = True  # true if the largest date must be actualized
        max_date = DailyQueryBatch.latest_day_available()
        max_overlap_left = min(len(df_a), overlap)  # upper bound considered for the overlap
        max_overlap_right = min(len(df_b), overlap)
        for df_candidate in list_df_daily:
            intersection_left = len(df_a.index.intersection(df_candidate.index))
            intersection_right = len(df_b.index.intersection(df_candidate.index))
            inter = min(intersection_left, intersection_right)
            if inter >= best_inter:
                if ((intersection_left == best_inter_left and intersection_right < best_inter_right) or
                        (intersection_right == best_inter_right and intersection_left < best_inter_left)):
                    continue
                best_df = df_candidate
                best_inter = inter
                best_inter_left = intersection_left
                best_inter_right = intersection_right
                if intersection_right < max_overlap_right and df_candidate.index.max() < max_date:  # new data is available
                    can_be_actualized = True
                elif intersection_left < max_overlap_left:  # better data can be found
                    can_be_actualized = True
                else:
                    can_be_actualized = False
        return best_df, can_be_actualized

    @staticmethod
    def find_min_dates_queries(list_dates: List[Tuple[datetime, datetime]], number: int) \
            -> List[Tuple[datetime, datetime]]:
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
                    largest_begin, largest_end = DailyQueryBatch.largest_dates_iterator(begin, end, number)
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

            def return_best_dates(list_node: Dict[int, List[Node]]):
                dates = []
                depth_covered = False
                depth = len(list_node) - 1
                while depth >= 0 and not depth_covered:
                    depth_covered = True
                    for node in list_node[depth]:
                        if node.feasible and not node.covered:
                            node.set_covered()
                            dates.append((node.begin, node.end))
                        elif not node.covered:
                            depth_covered = False
                    depth -= 1
                return dates

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
                for node_a, node_b in zip(list_node[depth - 1], list_node[depth - 1][1:]):
                    node_cur = Node(node_a.begin, node_b.end)
                    list_node[depth].append(node_cur)
                    node_a.add_child(node_cur)
                    node_b.add_child(node_cur)
            # retrieve the best interval by starting on the node at the largest depth
            # best_dates = return_best_date(list_node[len(list_dates) - 1][0])
            best_dates = return_best_dates(list_node)
            return best_dates


class MinimalQueryList(QueryList):
    """
    list of queries using the minimal number of requests.
    No mean, only one max daily query is used for retrieving a new topic
    New data is then actualized using hourly queries
    """

    def __init__(self, topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime, query_limit: int = 0, overlap: int = 30, number: int = 1,
                 cat: int = 0, gprop: str = '', savefile: bool = True, shuffle: bool = True):
        self.directory_hourly = dir_min_hourly
        self.query_hourly = HourlyQuery
        self.overlap_hourly_daily = 6  # overlap used for daily to hourly requests
        self.overlap_hourly = 15  # overlap used between 2 hourly requests
        self.part = 'daily'  # switch to 'hourly' once all daily requests have been processed
        self.delete_hourly = set()  # list of hourly files that must be deleted
        super().__init__(topics, geo, dir_min_daily, trends_request, begin, end, DailyQuery, query_limit, overlap,
                         number, cat, gprop, savefile, shuffle)

    def __call__(self, *args, **kwargs) -> bool:
        result = super().__call__()
        if result and self.part == 'daily':  # the last daily query has been done, processing for hourly queries
            self.query = self.query_hourly
            self.growth_iterator = self.growth_list_iterator_hourly()
            self.part = 'hourly'
            self.status = 'running'
            return False
        else:
            return result

    def growth_list_iterator(self) -> Iterator[Query]:
        """
        iterator used for the daily requests
        """
        self.status = 'running'
        for topic_name, topic_code in self.topics.items():
            for geo_code, geo_name in self.geo.items():
                filename = f'{self.directory_hourly}/{geo_code}-{topic_name}.csv'
                if os.path.exists(filename):
                    df_hourly = pd.read_csv(filename, parse_dates=['date'], date_parser=date_parser_hourly).set_index(
                        'date')
                    # check if the hourly data covered the last end asked. If yes, keep it. Otherwise, will be deleted
                    if df_hourly.index.max().to_pydatetime() < self.end:  # throw away the hourly data
                        end = self.end
                        self.delete_hourly.add(filename)
                    else:  # keep the hourly data
                        # first hourly date registered + timedelta
                        begin_hourly = (df_hourly.index.min() - timedelta(
                            days=(self.overlap_hourly_daily - 1))).to_pydatetime().date()
                        begin_hourly = date_to_datetime(begin_hourly)
                        end = min(self.end, begin_hourly)
                else:
                    end = self.end
                yield self.query(topic_name, topic_code, geo_code, self.trends_request, self.begin, end,
                                 self.directory, self.overlap, self.cat, self.number, self.gprop,
                                 savefile=self.savefile, shuffle=self.shuffle)
        self.status = 'done'
        yield None

    def growth_list_iterator_hourly(self) -> Iterator[Query]:
        """
        iterator used for the hourly requests. Only called once the daily requests have been processed
        """
        self.status = 'running'
        for topic_name, topic_code in self.topics.items():
            for geo_code, geo_name in self.geo.items():
                filename = f'{self.directory}/{geo_code}-{topic_name}.csv'
                filename_hourly = f'{self.directory_hourly}/{geo_code}-{topic_name}.csv'
                if filename_hourly in self.delete_hourly:  # need to actualise the hourly data col
                    if self.verbose:
                        print(f'removing outdated hourly file ({filename_hourly})')
                    os.remove(filename_hourly)
                if os.path.exists(filename):
                    df_daily = pd.read_csv(filename, parse_dates=['date'], date_parser=date_parser_daily).set_index(
                        'date')
                    # first hourly date registered + timedelta
                    end_daily = (df_daily.index.max() - timedelta(days=(self.overlap_hourly_daily - 1))).to_pydatetime()
                    begin = date_to_datetime(end_daily)
                    yield self.query_hourly(topic_name, topic_code, geo_code, self.trends_request, begin, self.end,
                                            self.directory_hourly, self.overlap_hourly, self.cat, self.number,
                                            self.gprop,
                                            savefile=self.savefile, shuffle=self.shuffle)
                else:
                    print(f'could not find file {filename}, ignoring hourly requests generation for it')
        self.status = 'done'
        yield None


class RelevantTopic:
    """
    get the most relevant topics that can be retrieved through pytrends, relative to the hospitalisations
    """

    def __init__(self, df_hospi, max_lag, threshold, max_steps,
                 init_topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime,
                 cat: int = 0, gprop: str = ''):
        assert len(geo) == 1
        assert (end - begin).days + 1 <= DailyQueryBatch.max_len()
        self.df_hospi = df_hospi
        self.init_topics = init_topics
        self.geo = [loc for loc in geo.keys()][0]
        self.max_lag = max_lag
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.trends_request = trends_request
        self.list_queries_interest = []
        self.new_df = {}
        self.begin = begin
        self.end = end
        self.correlation_df = pd.DataFrame(columns=("Topic_title", "Topic_mid", "Best delay", "Best correlation"))
        self.new_best_topics = []  # list of topics to use to generate related topics from
        self.related_topics = init_topics  # the first related topics are the initial topics
        self.new_related_topics = init_topics
        self.directory = dir_explore
        self.cat = cat
        self.gprop = gprop
        self.timeframe = dates_to_timeframe(self.begin, self.end, False)
        self.status = 'interest'  # one between 'interest' and 'related'
        self.growth_interest()

    @staticmethod
    def compute_corr(left, delta, hospi, init_df):
        """
        Computes the correlation between a particular feature and another
        :param left: feature that we want to correlate
        :param delta: delay introduced in the dataframe (in days)
        :param full_data_be: dataframe containing all the data
        :param init_df: initial dataframe
        :return: the correlation between the left feature and the number of new hospitalization
        """
        thisdata = pd.concat([init_df[left].shift(delta), hospi["TOT_HOSP"]], axis=1)
        return thisdata.corr()["TOT_HOSP"][left]

    def __call__(self, *args, **kwargs) -> bool:
        """
        process a query
        :return bool if all queries have been processed
        """
        if self.status == 'interest':
            return self.get_interest()
        elif self.status == 'related':
            return self.get_related()

    def get_interest(self) -> bool:
        if self.list_queries_interest:  # there are queries awaiting
            query = self.list_queries_interest.pop()
            query_finished = query()
            assert query_finished  # the queries should cover only one timeslot
            df = query.get_df()
            df = df.rolling(7, center=True).mean().dropna()
            self.new_df[query.topic_name] = df[[query.topic_code]]
            return False
        else:  # no more queries in the list
            self.steps += 1
            self.actualize_correlation()
            if self.steps >= self.max_steps:
                self.save_correlation()
                return True
            else:
                self.status = 'related'
                self()

    def get_related(self) -> bool:
        if self.new_best_topics:
            topic_code = self.new_best_topics[-1]
            # dict of topic_name, topic_code
            new_related = self.trends_request.related_topics([topic_code], self.cat, self.timeframe, self.geo,
                                                             self.gprop)
            # add only the new topics
            self.new_related_topics.update({k: v for k, v in new_related.items() if k not in self.related_topics})
            # topic saved for further usage
            self.related_topics.update(new_related)
            self.new_best_topics.pop()
            return False
        else:
            self.growth_interest()
            self.status = 'interest'
            self()

    def actualize_correlation(self):
        # compute the correlation
        data = []
        for topic_name, df in self.new_df.items():
            topic_code = df.columns[0]
            correlations = [(delay, self.compute_corr(topic_code, delay, self.df_hospi, df)) for delay in
                            range(0, self.max_lag)]
            best_delay, best_corr = max(correlations, key=lambda x: abs(x[1]))
            data.append([topic_name, topic_code, best_delay, best_corr])
        new_correlation = pd.DataFrame(data, columns=("Topic_title", "Topic_mid", "Best delay", "Best correlation"))
        self.correlation_df = self.correlation_df.append(new_correlation)  # actualize the correlation
        self.new_df = {}  # reset the new df obtained
        # new topics used to generate related topics
        self.new_best_topics = [val[1] for val in data if abs(val[3]) >= self.threshold]
        print('new promising topics:', self.new_best_topics)

    def growth_interest(self):
        for topic_name, topic_code in self.new_related_topics.items():
            self.list_queries_interest.append(DailyQuery(topic_name, topic_code, self.geo, self.trends_request,
                                                         self.begin, self.end, self.directory, 0, self.cat, 1,
                                                         self.gprop,
                                                         savefile=True, shuffle=True))
        self.new_related_topics = {}

    def save_correlation(self):
        self.correlation_df.drop_duplicates(subset='Topic_title', keep='first',
                                            inplace=True)  # drop duplicates in the dataframe
        self.correlation_df.reset_index(0, inplace=True, drop=True)  # reset the index --> index = date
        self.correlation_df = self.correlation_df.sort_values(by='Best correlation', key=lambda col: abs(col),
                                                              ascending=False)
        self.correlation_df.to_csv(f'{dir_explore}/{self.geo}-related_topics.csv',
                                   index=False)  # write results in a csv file

    def get_correlation(self):
        return self.correlation_df

    def set_trends_request(self, trends_request: TrendsRequest):
        """
        set the trends request for this instance, all its current queries and its newest queries
        :param trends_request: TrendsRequest instance to use
        """
        self.trends_request = trends_request
        for query in self.list_queries_interest:
            query.set_trends_request(trends_request)


class ModelData:  # base class used to generate model data

    def __init__(self, topics: Dict[str, str] = None, geo: Dict[str, str] = None):
        """
        :param topics: topics for which model data should be generated. None = all topics should be considered
        :param geo: geo for which model data should be generated. None = all loc should be considered
        """
        self.directory_model = dir_model
        self.topics = {} if topics is None else topics
        self.geo = {} if geo is None else geo

    def generate_model_data(self, savefile=True) -> Dict:  # generate model data using available information
        raise NotImplementedError

    @staticmethod
    def merge_trends_batches(left: pd.DataFrame, right: pd.DataFrame, topic: str, verbose: bool = False,
                             drop: Union[str, None] = None) -> pd.DataFrame:
        """
        return the concatenation of left and right, correctly scaled based on their overlap (in hours or days)
        :param left: accumulator dataframe
        :param right: new dataframe to add to the accumulator
        :param topic: topic code considered
        :param verbose: True if information must be printed
        :param drop: can be one of None, 'left' or 'right'. The function can choose to keep scaled values
            from left or right on the intersection. If drop == left -> keep the right values, if drop == right ->
            keep the left values. Else take whatever value.
        """
        if left.empty:
            return right
        # retrieve the overlapping points:
        intersection = left.index.intersection(right.index)
        left_overlap = left.loc[intersection]
        right_overlap = right.loc[intersection]
        # drop 0 and inf values on the overlap
        scaling_full = (left_overlap[topic] / right_overlap[topic]).replace([0, np.inf, -np.inf], np.nan).dropna()
        if verbose:
            print('overlap left:', left_overlap[topic])
            print('overlap right:', right_overlap[topic])
            print('ratio:', scaling_full)
            print("my drop=", drop)
        scaling = scaling_full.mean()
        if np.isnan(scaling):  # no valid data was found on the whole intersection, return zero dataframe
            left_to_keep = left[left.index < intersection.min()]
            scaled = left_to_keep.append(right)
            scaled[topic] = 0
        else:  # valid data was found on the intersection
            if scaling < 1:  # right has not the good scale
                if drop == 'left':
                    left_to_keep = left[left.index < intersection.min()]
                    right_to_add = right * scaling
                    scaled = left_to_keep.append(right_to_add)
                else:
                    right_to_add = right[right.index > intersection.max()]
                    right_to_add = right_to_add * scaling
                    scaled = left.append(right_to_add)
            else:  # left has not the good scale
                if drop == 'right':
                    right_to_keep = right[right.index > intersection.max()]
                    left_to_add = left / scaling
                    scaled = left_to_add.append(right_to_keep)
                else:
                    left_to_add = left[left.index < intersection.min()]
                    left_to_add = left_to_add / scaling
                    scaled = left_to_add.append(right)
        return scaled

    @staticmethod
    def scale_df(df: pd.DataFrame, topic) -> List[pd.DataFrame]:
        """
        Return a list of the scaled df. If there is always an overlap and the batch ids are valid, the list contains
        one df. Otherwise, the list contains as many df as there are clusters of periods without missing data
        :param df: pandas dataframe to scale. Must contain a 'batch_id' column
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
                scaled_df = ModelData.merge_trends_batches(scaled_df, batch_df, topic)
            elif (overlap_left[topic] * overlap_right[topic]).sum() == 0:  # cannot perform the merge
                list_scaled_df.append(scaled_df)
                scaled_df = batch_df
            else:
                scaled_df = ModelData.merge_trends_batches(scaled_df, batch_df, topic)
        list_scaled_df.append(scaled_df)

        return list_scaled_df


class DailyModelData(ModelData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = dir_daily

    def generate_model_data(self, savefile=True) -> Dict:
        """
        generate model data for the available queries, using the daily requests stored
        """
        models = {}
        for (dirpath, dirnames, filenames) in os.walk(self.directory):
            for filename in filenames:
                try:
                    search_obj = re.search('(.*)-([^-]*).csv', filename)
                    geo, topic_name = search_obj.group(1), search_obj.group(2)
                    if self.geo and self.topics and ((geo not in self.geo) or (topic_name not in self.topics)):
                        # the model data was not asked for this loc and this topic
                        continue
                    df_tot = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                         date_parser=date_parser_daily).set_index('date')
                    gb = df_tot.groupby("batch_id")
                    topic_code = df_tot.columns[-2]  # last column is batch_id, the one before is the topic code
                    list_df = [gb.get_group(x)[[topic_code]] for x in gb.groups]
                    model = list_df[0]
                    for df in list_df[1:]:
                        model = ModelData.merge_trends_batches(model, df, topic_code)
                    if geo in models:
                        models[geo][topic_name] = model
                    else:
                        models[geo] = {topic_name: model}
                    if savefile:
                        model.to_csv(f"{self.directory_model}/{filename}.csv")
                except (KeyError, AttributeError):
                    print(f'error when generating model data for {filename}')
        return models


class HourlyModelData(ModelData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory_hourly = dir_hourly
        self.directory_daily = dir_daily_gap

    def generate_model_data(self, savefile=True) -> Dict:
        """
        generate model data for the available queries, using the hourly requests stored and the daily gap requests stored
        for intervals where no data existed
        """
        models = {}
        exceptions_topic = {topic: [0, 0, []] for topic in self.topics}
        for (dirpath, dirnames, filenames) in os.walk(self.directory_hourly):
            for filename in filenames:
                topic_code = None
                topic_name = None
                begin = None
                end = None
                geo = None
                model = None
                try:
                    search_obj = re.match('([^_]*)-([^-_]*).csv', filename)
                    if search_obj is None:
                        continue
                    geo, topic_name = search_obj.group(1), search_obj.group(2)
                    if self.geo and self.topics and ((geo not in self.geo) or (topic_name not in self.topics)):
                        continue
                    exceptions_topic[topic_name][1] += 1
                    df_hourly = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                            date_parser=date_parser_hourly).set_index('date')
                    topic_code = df_hourly.columns[-2]
                    list_df_hourly = HourlyModelData.hourly_to_list_daily(df_hourly, topic_code)
                    starting_pattern = f"{geo}-{topic_name}-"
                    existing_files = [filename for filename in os.listdir(self.directory_daily) if
                                      filename.startswith(starting_pattern)]
                    list_daily_df = [pd.read_csv(f"{self.directory_daily}/{file}", parse_dates=['date'],
                                                 date_parser=date_parser_daily).set_index('date')[[topic_code]]
                                     for file in existing_files]

                    complete_df = HourlyModelData.merge_hourly_daily(list_df_hourly, list_daily_df, topic_code,
                                                                     drop=True)
                    filename = f"{self.directory_model}/{geo}-{topic_name}.csv"
                    if (complete_df[topic_code] == 0).all() or complete_df.isnull().values.any():
                        print(f'error when generating model data from {filename}. Generating zero dataframe instead')
                        exceptions_topic[topic_name][0] += 1
                        exceptions_topic[topic_name][2].append(geo)
                        complete_df[topic_code] = 0
                    if geo in models:
                        models[geo][topic_name] = complete_df
                    else:
                        models[geo] = {topic_name: complete_df}

                    if savefile:
                        complete_df.to_csv(filename)
                except (KeyError, AttributeError, ValueError):
                    exceptions_topic[topic_name][0] += 1
                    exceptions_topic[topic_name][2].append(geo)
                    print(f'error when generating model data for {filename}')

        for topic_name, (err, tot, loc) in exceptions_topic.items():
            print(f'topic {topic_name}: {err}/{tot} error', end='s' if err != 0 else '')
            if err != 0:
                print(' (', end='')
                for i, region in enumerate(sorted(loc)):
                    print(region, end='' if i == err - 1 else ', ')
                print(')')
            else:
                print()
        return models

    @staticmethod
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
            hours_drop = 11  # 5  # consecutive values to check
            delta = timedelta(hours=hours_drop - 1)
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
            cur_df = cur_df * 100 / cur_df.max()
            # check if the resulting dataframe can be added
            if not cur_df.empty and (new_end - new_begin).days >= 2 and len(np.where(cur_df > 10)[0]) > 3:
                result.append(cur_df)
        return result

    @staticmethod
    def hourly_to_list_daily(df: pd.DataFrame, topic_code: str) -> List[pd.DataFrame]:
        """
        sanitize hourly data, transforming it to a list of daily data and removing missing values
        :param df: dataframe of hourly data on a trends topic, indexed by hourly dates
        :param topic_code: code for the topic
        :return list of data sanitized: missing values are removed, leading to dataframes with holes between one another
        """
        list_df_hourly = ModelData.scale_df(df, topic_code)  # scale the dataframe
        list_df_hourly = HourlyModelData.drop_incomplete_days(
            list_df_hourly)  # drop the incomplete days (check doc for details)
        list_df_hourly = [df.resample('D').mean() for df in list_df_hourly]  # aggregate to daily data
        for i in range(len(list_df_hourly)):
            list_df_hourly[i] = list_df_hourly[i] * 100 / list_df_hourly[i].max()
        return list_df_hourly

    @staticmethod
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

        if len(overlap) < overlap_min or len(overlap_right) < overlap_min:
            overlap = df_left.index.intersection(df_daily.index)
            daily_used = df_daily
        else:
            daily_used = daily_rolling

        if len(overlap) > overlap_max:
            overlap = overlap[-overlap_max:]
        title = daily_used.columns[0]
        daily_used.loc[daily_used[title] < 0.01, title] = 0
        batch_rescaled = ModelData.merge_trends_batches(df_left, daily_used[overlap.min():], topic_code,
                                                        verbose=False, drop=drop[0])
        batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
        overlap = batch_rescaled.index.intersection(df_right.index)
        if len(overlap) > overlap_max:
            overlap = overlap[:overlap_max]
        batch_rescaled = ModelData.merge_trends_batches(batch_rescaled[:overlap.max()], df_right, topic_code,
                                                        verbose=False, drop=drop[1])
        batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
        return batch_rescaled

    @staticmethod
    def merge_hourly_daily(list_df_hourly: List[pd.DataFrame], list_df_daily: List[pd.DataFrame], topic_code: str,
                           drop: bool, add_daily_end=True):
        """
        merge the hourly (deterministic) aggregated batches, using the daily (stochastic) batches on the missing interval
        raise a value error if it is impossible to perform the merge
        :param list_df_hourly: sorted list of deterministic DataFrame, having a daily index
        :param list_df_daily: list of stochastic DataFrame, having data on the missing interval of list_df_hourly
        :param topic_code: topic code
        :param drop: whether to drop the stochastic data preferably or not
        :param add_daily_end: if True, add daily data data at the end if the max date of daily data > max date of hourly data
        :return df: merged DataFrame
        """
        df = list_df_hourly[0]
        # attempt to add at the beginning if possible
        daily_possible = [df_daily for df_daily in list_df_daily if df_daily.index.min() < df.index.min()]
        if len(daily_possible) != 0:
            column = df.columns[0]
            candidate = max(daily_possible, key=lambda df_daily: df_daily.index.intersection(df.index))
            overlap_len = len(df.index.intersection(candidate.index))
            if overlap_len != 0:  # possible to add the data since there is an overlap
                df = ModelData.merge_trends_batches(candidate, df, column, verbose=False, drop='left')
                df = df * 100 / df.max()

        for df_right in list_df_hourly[1:]:
            df_daily, _ = DailyGapQueryList.find_largest_intersection(df, df_right, list_df_daily)
            if df_daily.empty:
                raise ValueError('no valid dataframe found on the gap')
            df_1 = HourlyModelData.rescale_batch(df, df_right, df_daily, topic_code, drop=drop)
            df = df_1

        if add_daily_end:  # attempt to add daily data at the end
            daily_possible = [df_daily for df_daily in list_df_daily if df_daily.index.max() > df.index.max()]
            if len(daily_possible) != 0:
                column = df.columns[0]
                while len(daily_possible) > 0:
                    candidate = max(daily_possible, key=lambda df_daily: df_daily.index.intersection(df.index))
                    overlap_len = len(df.index.intersection(candidate.index))
                    if overlap_len == 0:  # not possible to add the data since there is not overlap
                        break
                    df = ModelData.merge_trends_batches(df, candidate, column, verbose=False, drop='right')
                    df = df * 100 / df.max()
                    daily_possible = [df_daily for df_daily in daily_possible if df_daily.index.max() > df.index.max()]
        return df


class MinimalModelData(ModelData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory_hourly = dir_min_hourly
        self.directory_daily = dir_min_daily

    def generate_model_data(self, savefile=True) -> Dict:
        """
        generate model data for the available queries, using the hourly requests stored and the daily gap requests stored
        for intervals where no data existed
        """
        # list of [exceptions caught, files considered, [regions exceptions]]
        exceptions_topic = {topic: [0, 0, []] for topic in self.topics}
        models = {}
        for (dirpath, dirnames, filenames) in os.walk(self.directory_hourly):
            for filename in filenames:
                topic_code = None
                topic_name = None
                begin = None
                end = None
                geo = None
                model = None
                try:
                    search_obj = re.match('([^_]*)-([^-_]*).csv', filename)
                    if search_obj is None:
                        continue
                    geo, topic_name = search_obj.group(1), search_obj.group(2)
                    if self.geo and self.topics and ((geo not in self.geo) or (topic_name not in self.topics)):
                        continue
                    exceptions_topic[topic_name][1] += 1
                    daily_df = pd.read_csv(f"{self.directory_daily}/{filename}", parse_dates=['date'],
                                           date_parser=date_parser_daily).set_index('date')
                    gb = daily_df.groupby("batch_id")
                    topic_code = daily_df.columns[-2]  # last column is batch_id, the one before is the topic code
                    list_df = [gb.get_group(x)[[topic_code]] for x in gb.groups]

                    df_hourly = pd.read_csv(f'{dirpath}/{filename}', parse_dates=['date'],
                                            date_parser=date_parser_hourly).set_index('date')
                    begin = daily_df.index.min().to_pydatetime()
                    end = df_hourly.index.max().to_pydatetime()
                    topic_code = df_hourly.columns[-2]
                    list_df_hourly = HourlyModelData.hourly_to_list_daily(df_hourly, topic_code)

                    model = list_df[0]
                    for df in list_df[1:]:  # add the daily data
                        model = ModelData.merge_trends_batches(model, df, topic_code)
                    # add hourly data at the end
                    model = ModelData.merge_trends_batches(model, list_df_hourly[0], topic_code)
                    filename_model = f"{self.directory_model}/{geo}-{topic_name}.csv"
                    if (model[topic_code] == 0).all():
                        print(f'error when generating model data from {filename}. Generating zero dataframe instead')
                        exceptions_topic[topic_name][0] += 1
                        exceptions_topic[topic_name][2].append(geo)
                    if savefile:
                        model.to_csv(filename_model)
                except (KeyError, AttributeError, ValueError, IndexError):
                    print(f'error when generating model data from {filename}', end='')
                    if end is not None:
                        print('. Generating zero dataframe instead')
                        model = TrendsRequest.zero_dataframe([topic_code], begin, end).drop(columns=['isPartial'])
                        filename_model = f"{self.directory_model}/{geo}-{topic_name}.csv"
                        if savefile:
                            model.to_csv(filename_model)
                    else:
                        model = pd.DataFrame()
                        print()
                    exceptions_topic[topic_name][0] += 1
                    exceptions_topic[topic_name][2].append(geo)
                finally:
                    if geo in models:
                        models[geo][topic_name] = model
                    else:
                        models[geo] = {topic_name: model}
        for topic_name, (err, tot, loc) in exceptions_topic.items():
            print(f'topic {topic_name}: {err}/{tot} error', end='s' if err != 0 else '')
            if err != 0:
                print(' (', end='')
                for i, region in enumerate(sorted(loc)):
                    print(region, end='' if i == err - 1 else ', ')
                print(')')
            else:
                print()
        return models


def run_collect(trends_request, query_list):
    """
    run a callable (QueryList or similar: callable returning True when finished) several time until it is finished.
    Switch to Tor if an exception is thrown
    :param trends_request: trends request instance to use for the collection of data
    :param collection: QueryList instance / callable instance to use. Must return False when it needs to be processed
        and True once it is done
    :return: trends_request instance used at the end of the iterations
    """
    finished = False
    time_init = time.perf_counter()

    def time_for_change_fun():
        return random.randint(5000, 5500)

    def time_sleep_change():
        return random.randint(540, 900)

    time_for_change = time_for_change_fun()
    while not finished:
        try:
            finished = query_list()
        except google_trends_errors as err:  # trends_request failed, using tor for the remaining requests
            if tor_ip_set:
                print('using tor')
                trends_request = TorTrendsRequest(max_errors=np.inf)
            else:
                trends_request = LocalTrendsRequest(max_errors=np.inf)
            query_list.set_trends_request(trends_request)
        time_current = time.perf_counter()
        elapsed = time_current - time_init
        if elapsed > time_for_change:
            time_init = time_current
            time_for_change = time_for_change_fun()
            if isinstance(trends_request, TorTrendsRequest):
                trends_request.pytrends = None
            time.sleep(time_sleep_change())
            if isinstance(trends_request, TorTrendsRequest):
                trends_request.pytrends = trends_request.get_new_ip()
    return trends_request


def collect_data(method: str = 'daily', topics: Dict[str, str] = None, geo: Dict[str, str] = None, gap: bool = True):
    """
    collect the google trends data
    :param daily: whether to collect the data using the daily method or not
    :param topics: dict of topic_name, topic_code to query. if None, uses util util.list_topics
    :param geo: dict of geo_code, geo_name to query. if None, uses util.french_region_and_be
    :param gap: if True and daily is False, collect the data for the gap in the hourly queries
    :return: None. save the results to the data/trends folder
    """
    if topics is None:
        topics = util.list_topics
    if geo is None:
        geo = util.french_region_and_be

    trends_request = LocalTrendsRequest(max_errors=4)  # use local queries for the beginning
    begin = datetime.strptime('2020-02-01', '%Y-%m-%d')
    if method == 'daily':
        end = DailyQueryBatch.latest_day_available()
        query_list = DailyQueryList(topics, geo, trends_request, begin, end, number=10, savefile=True)
    elif method == 'hourly':
        end = datetime.strptime('2021-05-31T23', hour_format)
        query_list = HourlyQueryList(topics, geo, trends_request, begin, end, number=1, savefile=True)
    elif method == 'minimal':
        end = HourlyQueryBatch.latest_day_available()
        query_list = MinimalQueryList(topics, geo, trends_request, begin, end, savefile=True)
    elif method == 'gap':
        end = datetime.strptime('2021-05-31', day_format)
        query_list = DailyGapQueryList(topics, geo, trends_request, begin, end, number=10, savefile=True, shuffle=False)
    trends_request = run_collect(trends_request, query_list)


def collect_minimal_data(topics: Dict[str, str] = None, geo: Dict[str, str] = None):
    if topics is None:
        topics = util.list_topics
    if geo is None:
        geo = util.french_region_and_be
    # trend_request = FakeTrendsRequest()  # use local queries for the beginning
    trends_request = LocalTrendsRequest(max_errors=2)  # use local queries for the beginning
    begin = datetime.strptime('2020-02-01', day_format)
    end = HourlyQueryBatch.latest_day_available()
    query_list = MinimalQueryList(topics, geo, trends_request, begin, end, savefile=True)
    run_collect(trends_request, query_list)


def collect_relevant_topics(topics: Dict[str, str] = None, geo: Dict[str, str] = None):
    if topics is None:
        topics = util.list_topics
    if geo is None:
        geo = {'BE': 'Belgium'}
    trends_request = LocalTrendsRequest(max_errors=2)  # use local queries for the beginning
    begin = datetime.strptime('2021-02-01', day_format)
    end = datetime.strptime('2021-05-14', day_format)
    '''
        def __init__(self, df_hospi, max_lag, threshold, max_steps,
                 init_topics: Dict[str, str], geo: Dict[str, str], trends_request: TrendsRequest,
                 begin: datetime, end: datetime,
                 cat: int = 0, gprop: str = ''):
     '''
    url_hospi_belgium = "../data/hospi/be-covid-hospi.csv"
    url_department_france = "france_departements.csv"
    url_hospi_france_new = "../data/hospi/fr-covid-hospi.csv"
    url_hospi_france_tot = "../data/hospi/fr-covid-hospi-total.csv"
    df_hospi = util.hospi_french_region_and_be(url_hospi_france_tot, url_hospi_france_new, url_hospi_belgium,
                                               url_department_france, util.french_region_and_be, new_hosp=True,
                                               tot_hosp=True)['BE']
    df_hospi = df_hospi.reset_index().rename(columns={'DATE': 'date'}).drop(columns=['LOC']).set_index('date'). \
        rolling(7, center=True).mean().dropna()
    queries = RelevantTopic(df_hospi, 25, 0.75, 5, topics, geo, trends_request, begin, end)
    run_collect(trends_request, queries)


def generate_model_data(method: str = 'daily', topics: Dict[str, str] = None, geo: Dict[str, str] = None,
                        savefile=True) -> Dict:
    """
    generate model data
    :param daily: whether to generate the model data using the daily method or not
    :return: Dict of {geo_code: {topic: df}} generated. save the results to the data/trends/model folder
    """
    if method == 'daily':
        model_data = DailyModelData(topics, geo)
    elif method == 'hourly':
        model_data = HourlyModelData(topics, geo)
    elif method == 'minimal':
        model_data = MinimalModelData(topics, geo)
    else:
        raise Exception(f'unimplemented model generation ({method})')
    models = model_data.generate_model_data(savefile)
    return models


def collect_for_plots():
    begin = datetime.strptime('2020-06-01', day_format)
    end = datetime.strptime('2020-09-30', day_format)
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topics = {topic_name: topic_code}
    geo_code, geo_name = 'BE', 'Belgium'
    geo = {geo_code: geo_name}
    trends_request = LocalTrendsRequest(max_errors=2)  # use local queries for the beginning
    query_batch = DailyQueryBatch(topic_code, geo_code, trends_request, begin, end, 1, number=1)
    run_collect(trends_request, query_batch)
    df = query_batch.get_df()
    df = df.drop(columns=['batch_id'])
    df.to_csv(f'../data/trends/samples/{geo_code}-{topic_name}_daily_right.csv')


def nb_request_1_year():
    begin = datetime.strptime('2020-02-01', day_format)
    end = datetime.strptime('2021-02-01', day_format)
    topic_name, topic_code = 'Fièvre', '/m/0cjf0'
    topics = {topic_name: topic_code}
    geo_code, geo_name = 'US', 'United States'
    geo = {geo_code: geo_name}
    trends_request = LocalTrendsRequest(max_errors=2)  # use local queries for the beginning
    dummy = HourlyQuery(topic_name, topic_code, geo_code, trends_request, begin, end, dir_hourly, overlap=15)
    print(len(dummy.list_dates()))


if __name__ == '__main__':
    topics = util.list_topics_fr_hourly
    geo = util.french_region_and_be
    mode = 'minimal'
    collect_data(mode, topics, geo)
    generate_model_data(mode, topics, geo)
    # collect_relevant_topics()
    # collect_for_plots()
