from __future__ import absolute_import, print_function, unicode_literals
from time import sleep, time
from pytrends.exceptions import ResponseError
import os.path

#from my_fake_useragent import UserAgent
import random
from requests.exceptions import ReadTimeout
from urllib.parse import quote
from stem import Signal
from stem.control import Controller
from pandas.io.json._normalize import nested_to_record
from requests.packages.urllib3.util.retry import Retry
from .filter import filter_family, filter_os_family, filter_phone, filter_version_range
from .utils import build_stream_function
from pytrends import exceptions
import json
from datetime import datetime, timedelta

import pandas as pd
import requests

from pytrends import exceptions

class UserAgent():
    from my_fake_useragent.parsed_data import parsed_data

    def __init__(self,
                 family=None,
                 os_family=None,
                 phone=None,
                 version_range=None,
                 ):
        """
        :param mode: default mode
        :param family: 不设置则不管 指定浏览器类型
        :param os_family: 不设置则不管 指定操作系统
        :param phone: 指定是否是手机端 True 是 False 不是 不设置默认None则不管
        :param version_range: 不设置则不管 指定浏览器版本范围
        手机检测 根据设备family参数之外 操作系统检测到 android 或 ios 也认为是移动端
        """

        if isinstance(family, str):
            family = family.lower()
            self.family = [family]
        elif isinstance(family, (list, tuple)):
            self.family = [f.lower() for f in family]
        elif family is None:
            self.family = None
        else:
            raise ValueError('family')

        if isinstance(os_family, str):
            os_family = os_family.lower()
            self.os_family = [os_family]
        elif isinstance(os_family, (list, tuple)):
            self.os_family = [f.lower() for f in os_family]
        elif os_family is None:
            self.os_family = None
        else:
            raise ValueError('os_family')

        self.phone = phone
        if self.phone not in [None, True, False]:
            raise ValueError('phone')

        self.version_range = version_range

        self.filter_func = build_stream_function(filter_family,
                                                 filter_os_family, filter_phone,
                                                 filter_version_range)

    def random(self):
        user_agent_list = self.get_useragent_list()

        if user_agent_list:
            return random.choice(user_agent_list)
        else:
            raise Exception('empty result')

    def get_useragent_list(self):
        origin_data = []
        for key in self.parsed_data:
            origin_data += self.parsed_data[key]

        d = {
            'data': origin_data,
            'family': self.family,
            'version_range': self.version_range,
            'os_family': self.os_family,
            'phone': self.phone
        }

        d = self.filter_func(d)

        ua_string_list = [i['string'] for i in d['data']]
        return ua_string_list

    def test_possible_family(self):
        t1 = set()
        for k, v in self.parsed_data.items():
            for i in v:
                t1.add(i['user_agent']['family'])
        return t1

    def test_possible_os_family(self):
        t1 = set()
        for k, v in self.parsed_data.items():
            for i in v:
                t1.add(i['os']['family'])
        return t1

    def test_possible_device_family(self):
        t1 = set()
        for k, v in self.parsed_data.items():
            for i in v:
                t1.add(i['device']['family'])
        return t1


class TrendReq(object):
    """
    Google Trends API
    """
    GET_METHOD = 'get'
    POST_METHOD = 'post'
    GENERAL_URL = 'https://trends.google.com/trends/api/explore'
    INTEREST_OVER_TIME_URL = 'https://trends.google.com/trends/api/widgetdata/multiline'
    INTEREST_BY_REGION_URL = 'https://trends.google.com/trends/api/widgetdata/comparedgeo'
    RELATED_QUERIES_URL = 'https://trends.google.com/trends/api/widgetdata/relatedsearches'
    TRENDING_SEARCHES_URL = 'https://trends.google.com/trends/hottrends/visualize/internal/data'
    TOP_CHARTS_URL = 'https://trends.google.com/trends/api/topcharts'
    SUGGESTIONS_URL = 'https://trends.google.com/trends/api/autocomplete/'
    CATEGORIES_URL = 'https://trends.google.com/trends/api/explore/pickers/category'
    TODAY_SEARCHES_URL = 'https://trends.google.com/trends/api/dailytrends'

    def __init__(self, hl='en-US', tz=360, geo='', timeout=(2, 5), proxies='',
                 retries=0, backoff_factor=0, requests_args=None, custom_useragent=None):
        """
        Initialize default values for params
        """
        # google rate limit
        self.google_rl = 'You have reached your quota limit. Please try again later.'
        self.results = None
        # set user defined options used globally
        self.tz = tz
        self.hl = hl
        self.geo = geo
        self.kw_list = list()
        self.timeout = timeout
        self.proxies = proxies  # add a proxy option
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.proxy_index = 0
        self.url_login = "https://accounts.google.com/ServiceLogin"
        self.url_auth = "https://accounts.google.com/ServiceLoginAuth"
        self.requests_args = requests_args or {}
        self.cookies = self.GetGoogleCookie()
        # intialize widget payloads
        self.token_payload = dict()
        self.interest_over_time_widget = dict()
        self.interest_by_region_widget = dict()
        self.related_topics_widget_list = list()
        self.related_queries_widget_list = list()
        if custom_useragent is None:
            self.custom_useragent = {'User-Agent': 'PyTrends'}
        else:
            self.custom_useragent = {'User-Agent': custom_useragent}

    def GetGoogleCookie(self):
        """
        Gets google cookie (used for each and every proxy; once on init otherwise)
        Removes proxy from the list on proxy error
        """
        while True:
            if len(self.proxies) > 0:
                proxy = {'https': self.proxies[self.proxy_index]}
            else:
                proxy = ''
            try:
                return dict(filter(lambda i: i[0] == 'NID', requests.get(
                    'https://trends.google.com/?geo={geo}'.format(
                        geo=self.hl[-2:]),
                    timeout=self.timeout,
                    proxies=proxy,
                    **self.requests_args
                ).cookies.items()))
            except requests.exceptions.ProxyError:
                print('Proxy error. Changing IP')
                if len(self.proxies) > 1:
                    self.proxies.remove(self.proxies[self.proxy_index])
                else:
                    print('No more proxies available. Bye!')
                    raise
                continue

    def GetNewProxy(self):
        """
        Increment proxy INDEX; zero on overflow
        """
        if self.proxy_index < (len(self.proxies) - 1):
            self.proxy_index += 1
        else:
            self.proxy_index = 0

    def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
        """Send a request to Google and return the JSON response as a Python object
        :param url: the url to which the request will be sent
        :param method: the HTTP method ('get' or 'post')
        :param trim_chars: how many characters should be trimmed off the beginning of the content of the response
            before this is passed to the JSON parser
        :param kwargs: any extra key arguments passed to the request builder (usually query parameters or data)
        :return:
        """
        s = requests.session()
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
                              cookies=self.cookies, **kwargs, **self.requests_args)  # DO NOT USE retries or backoff_factor here
        else:
            response = s.get(url, timeout=self.timeout, cookies=self.cookies,
                             **kwargs, **self.requests_args)   # DO NOT USE retries or backoff_factor here
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

    def build_payload(self, kw_list, cat=0, timeframe='today 5-y', geo='',
                      gprop=''):
        """Create the payload for related queries, interest over time and interest by region"""
        self.kw_list = kw_list
        self.geo = geo or self.geo
        self.token_payload = {
            'hl': self.hl,
            'tz': self.tz,
            'req': {'comparisonItem': [], 'category': cat, 'property': gprop}
        }

        # build out json for each keyword
        for kw in self.kw_list:
            keyword_payload = {'keyword': kw, 'time': timeframe,
                               'geo': self.geo}
            self.token_payload['req']['comparisonItem'].append(keyword_payload)
        # requests will mangle this if it is not a string
        self.token_payload['req'] = json.dumps(self.token_payload['req'])
        # get tokens
        self._tokens()
        return

    def _tokens(self):
        """Makes request to Google to get API tokens for interest over time, interest by region and related queries"""
        # make the request and parse the returned json
        widget_dict = self._get_data(
            url=TrendReq.GENERAL_URL,
            method=TrendReq.GET_METHOD,
            params=self.token_payload,
            trim_chars=4,
        )['widgets']
        # order of the json matters...
        first_region_token = True
        # clear self.related_queries_widget_list and self.related_topics_widget_list
        # of old keywords'widgets
        self.related_queries_widget_list[:] = []
        self.related_topics_widget_list[:] = []
        # assign requests
        for widget in widget_dict:
            if widget['id'] == 'TIMESERIES':
                self.interest_over_time_widget = widget
            if widget['id'] == 'GEO_MAP' and first_region_token:
                self.interest_by_region_widget = widget
                first_region_token = False
            # response for each term, put into a list
            if 'RELATED_TOPICS' in widget['id']:
                self.related_topics_widget_list.append(widget)
            if 'RELATED_QUERIES' in widget['id']:
                self.related_queries_widget_list.append(widget)
        return

    def interest_over_time(self):
        """Request data from Google's Interest Over Time section and return a dataframe"""

        over_time_payload = {
            # convert to string as requests will mangle
            'req': json.dumps(self.interest_over_time_widget['request']),
            'token': self.interest_over_time_widget['token'],
            'tz': self.tz
        }

        # make the request and parse the returned json
        req_json = self._get_data(
            url=TrendReq.INTEREST_OVER_TIME_URL,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            params=over_time_payload,
        )

        df = pd.DataFrame(req_json['default']['timelineData'])
        if (df.empty):
            return df

        df['date'] = pd.to_datetime(df['time'].astype(dtype='float64'),
                                    unit='s')
        df = df.set_index(['date']).sort_index()
        # split list columns into seperate ones, remove brackets and split on comma
        result_df = df['value'].apply(lambda x: pd.Series(
            str(x).replace('[', '').replace(']', '').split(',')))
        # rename each column with its search term, relying on order that google provides...
        for idx, kw in enumerate(self.kw_list):
            # there is currently a bug with assigning columns that may be
            # parsed as a date in pandas: use explicit insert column method
            result_df.insert(len(result_df.columns), kw,
                             result_df[idx].astype('int'))
            del result_df[idx]

        if 'isPartial' in df:
            # make other dataframe from isPartial key data
            # split list columns into seperate ones, remove brackets and split on comma
            df = df.fillna(False)
            result_df2 = df['isPartial'].apply(lambda x: pd.Series(
                str(x).replace('[', '').replace(']', '').split(',')))
            result_df2.columns = ['isPartial']
            # concatenate the two dataframes
            final = pd.concat([result_df, result_df2], axis=1)
        else:
            final = result_df
            final['isPartial'] = False

        return final

    def interest_by_region(self, resolution='COUNTRY', inc_low_vol=False,
                           inc_geo_code=False):
        """Request data from Google's Interest by Region section and return a dataframe"""

        # make the request
        region_payload = dict()
        if self.geo == '':
            self.interest_by_region_widget['request'][
                'resolution'] = resolution
        elif self.geo == 'US' and resolution in ['DMA', 'CITY', 'REGION']:
            self.interest_by_region_widget['request'][
                'resolution'] = resolution

        self.interest_by_region_widget['request'][
            'includeLowSearchVolumeGeos'] = inc_low_vol

        # convert to string as requests will mangle
        region_payload['req'] = json.dumps(
            self.interest_by_region_widget['request'])
        region_payload['token'] = self.interest_by_region_widget['token']
        region_payload['tz'] = self.tz

        # parse returned json
        req_json = self._get_data(
            url=TrendReq.INTEREST_BY_REGION_URL,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            params=region_payload,
        )
        df = pd.DataFrame(req_json['default']['geoMapData'])
        if (df.empty):
            return df

        # rename the column with the search keyword
        df = df[['geoName', 'geoCode', 'value']].set_index(
            ['geoName']).sort_index()
        # split list columns into seperate ones, remove brackets and split on comma
        result_df = df['value'].apply(lambda x: pd.Series(
            str(x).replace('[', '').replace(']', '').split(',')))
        if inc_geo_code:
            result_df['geoCode'] = df['geoCode']

        # rename each column with its search term
        for idx, kw in enumerate(self.kw_list):
            result_df[kw] = result_df[idx].astype('int')
            del result_df[idx]

        return result_df

    def related_topics(self):
        """Request data from Google's Related Topics section and return a dictionary of dataframes

        If no top and/or rising related topics are found, the value for the key "top" and/or "rising" will be None
        """

        # make the request
        related_payload = dict()
        result_dict = dict()
        for request_json in self.related_topics_widget_list:
            # ensure we know which keyword we are looking at rather than relying on order
            kw = request_json['request']['restriction'][
                'complexKeywordsRestriction']['keyword'][0]['value']
            # convert to string as requests will mangle
            related_payload['req'] = json.dumps(request_json['request'])
            related_payload['token'] = request_json['token']
            related_payload['tz'] = self.tz

            # parse the returned json
            req_json = self._get_data(
                url=TrendReq.RELATED_QUERIES_URL,
                method=TrendReq.GET_METHOD,
                trim_chars=5,
                params=related_payload,
            )

            # top topics
            try:
                top_list = req_json['default']['rankedList'][0][
                    'rankedKeyword']
                df_top = pd.DataFrame(
                    [nested_to_record(d, sep='_') for d in top_list])
            except KeyError:
                # in case no top topics are found, the lines above will throw a KeyError
                df_top = None

            # rising topics
            try:
                rising_list = req_json['default']['rankedList'][1][
                    'rankedKeyword']
                df_rising = pd.DataFrame(
                    [nested_to_record(d, sep='_') for d in rising_list])
            except KeyError:
                # in case no rising topics are found, the lines above will throw a KeyError
                df_rising = None

            result_dict[kw] = {'rising': df_rising, 'top': df_top}
        return result_dict

    def related_queries(self):
        """Request data from Google's Related Queries section and return a dictionary of dataframes

        If no top and/or rising related queries are found, the value for the key "top" and/or "rising" will be None
        """

        # make the request
        related_payload = dict()
        result_dict = dict()
        for request_json in self.related_queries_widget_list:
            # ensure we know which keyword we are looking at rather than relying on order
            kw = request_json['request']['restriction'][
                'complexKeywordsRestriction']['keyword'][0]['value']
            # convert to string as requests will mangle
            related_payload['req'] = json.dumps(request_json['request'])
            related_payload['token'] = request_json['token']
            related_payload['tz'] = self.tz

            # parse the returned json
            req_json = self._get_data(
                url=TrendReq.RELATED_QUERIES_URL,
                method=TrendReq.GET_METHOD,
                trim_chars=5,
                params=related_payload,
            )

            # top queries
            try:
                top_df = pd.DataFrame(
                    req_json['default']['rankedList'][0]['rankedKeyword'])
                top_df = top_df[['query', 'value']]
            except KeyError:
                # in case no top queries are found, the lines above will throw a KeyError
                top_df = None

            # rising queries
            try:
                rising_df = pd.DataFrame(
                    req_json['default']['rankedList'][1]['rankedKeyword'])
                rising_df = rising_df[['query', 'value']]
            except KeyError:
                # in case no rising queries are found, the lines above will throw a KeyError
                rising_df = None

            result_dict[kw] = {'top': top_df, 'rising': rising_df}
        return result_dict

    def trending_searches(self, pn='united_states'):
        """Request data from Google's Hot Searches section and return a dataframe"""

        # make the request
        # forms become obsolute due to the new TRENDING_SEACHES_URL
        # forms = {'ajax': 1, 'pn': pn, 'htd': '', 'htv': 'l'}
        req_json = self._get_data(
            url=TrendReq.TRENDING_SEARCHES_URL,
            method=TrendReq.GET_METHOD,
            **self.requests_args
        )[pn]
        result_df = pd.DataFrame(req_json)
        return result_df

    def today_searches(self, pn='US'):
        """Request data from Google Daily Trends section and returns a dataframe"""
        forms = {'ns': 15, 'geo': pn, 'tz': '-180', 'hl': 'en-US'}
        req_json = self._get_data(
            url=TrendReq.TODAY_SEARCHES_URL,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            params=forms,
            **self.requests_args
        )['default']['trendingSearchesDays'][0]['trendingSearches']
        result_df = pd.DataFrame()
        # parse the returned json
        sub_df = pd.DataFrame()
        for trend in req_json:
            sub_df = sub_df.append(trend['title'], ignore_index=True)
        result_df = pd.concat([result_df, sub_df])
        return result_df.iloc[:, -1]

    def top_charts(self, date, hl='en-US', tz=300, geo='GLOBAL'):
        """Request data from Google's Top Charts section and return a dataframe"""
        # create the payload
        chart_payload = {'hl': hl, 'tz': tz, 'date': date, 'geo': geo,
                         'isMobile': False}

        # make the request and parse the returned json
        req_json = self._get_data(
            url=TrendReq.TOP_CHARTS_URL,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            params=chart_payload,
            **self.requests_args
        )['topCharts'][0]['listItems']
        df = pd.DataFrame(req_json)
        return df

    def suggestions(self, keyword):
        """Request data from Google's Keyword Suggestion dropdown and return a dictionary"""

        # make the request
        kw_param = quote(keyword)
        parameters = {'hl': self.hl}

        req_json = self._get_data(
            url=TrendReq.SUGGESTIONS_URL + kw_param,
            params=parameters,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            **self.requests_args
        )['default']['topics']
        return req_json

    def categories(self):
        """Request available categories data from Google's API and return a dictionary"""

        params = {'hl': self.hl}

        req_json = self._get_data(
            url=TrendReq.CATEGORIES_URL,
            params=params,
            method=TrendReq.GET_METHOD,
            trim_chars=5,
            **self.requests_args
        )
        return req_json

    def get_historical_interest(self, keywords, year_start=2018, month_start=1,
                                day_start=1, hour_start=0, year_end=2018,
                                month_end=2, day_end=1, hour_end=0, cat=0,
                                geo='', gprop='', sleep=0):
        """Gets historical hourly data for interest by chunking requests to 1 week at a time (which is what Google allows)"""

        # construct datetime obejcts - raises ValueError if invalid parameters
        initial_start_date = start_date = datetime(year_start, month_start,
                                                   day_start, hour_start)
        end_date = datetime(year_end, month_end, day_end, hour_end)

        # the timeframe has to be in 1 week intervals or Google will reject it
        delta = timedelta(days=7)

        df = pd.DataFrame()

        date_iterator = start_date
        date_iterator += delta

        while True:
            # format date to comply with API call

            start_date_str = start_date.strftime('%Y-%m-%dT%H')
            date_iterator_str = date_iterator.strftime('%Y-%m-%dT%H')

            tf = start_date_str + ' ' + date_iterator_str

            try:
                self.build_payload(keywords, cat, tf, geo, gprop)
                week_df = self.interest_over_time()
                df = df.append(week_df)
            except Exception as e:
                print(e)
                pass

            start_date += delta
            date_iterator += delta

            if (date_iterator > end_date):
                # Run for 7 more days to get remaining data that would have been truncated if we stopped now
                # This is needed because google requires 7 days yet we may end up with a week result less than a full week
                start_date_str = start_date.strftime('%Y-%m-%dT%H')
                date_iterator_str = date_iterator.strftime('%Y-%m-%dT%H')

                tf = start_date_str + ' ' + date_iterator_str

                try:
                    self.build_payload(keywords, cat, tf, geo, gprop)
                    week_df = self.interest_over_time()
                    df = df.append(week_df)
                except Exception as e:
                    print(e)
                    pass
                break

            # just in case you are rate-limited by Google. Recommended is 60 if you are.
            if sleep > 0:
                time.sleep(sleep)

        # Return the dataframe with results from our timeframe
        return df.loc[initial_start_date:end_date]


ua = UserAgent()


def renew_tor_ip():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password="coucou2000")
        controller.signal(Signal.NEWNYM)


def collect_historical_interest(topic_mid, topic_title, geo, begin_tot=None, end_tot=None, overlap_hour=15, verbose=True):
    """
    load and collect hourly trends data for a given topic over a certain region

    :param topic_mid: mid code
    :param topic_title: title of the topic
    :param geo: google geocode
    :param begin_tot: beginning date. If None, default to first february
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
    if end_tot is None:
        end_tot = datetime.now()

    if os.path.exists(file):  # load previous file
        date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        df_tot = pd.read_csv(file, parse_dates=['date'], date_parser=date_parser).set_index('date')
        max_batch = df_tot[batch_column].max()
        if len(df_tot.loc[df_tot[batch_column] == max_batch]) < 192:  # if the last batch was not done on a whole week
            df_tot = df_tot.loc[df_tot[batch_column] < max_batch]  # drop the last batch
        i = df_tot[batch_column].max() + 1  # id of the next batch
        begin_tot = df_tot.index.max() - timedelta(hours=(overlap_hour-1))
        if end_tot - begin_tot < min_delta:  # must have a length of min 3 days
            begin_tot = end_tot - min_delta
    else:
        df_tot = pd.DataFrame()
        if begin_tot is None:
            begin_tot = datetime.strptime("2020-02-01T00", hour_format)
        i = 0

    begin_cur = begin_tot  # beginning of current batch
    end_cur = begin_tot + timedelta(days=7, hours=23)  # end of current batch
    delta = timedelta(days=7, hours=23) - timedelta(hours=(overlap_hour-1))  # diff between 2 batches
    delay = 0
    finished = False
    if verbose:
        print(f"topic {topic_title} geo {geo}")
    while not finished:
        try:
            sleep(5)
            timeframe = begin_cur.strftime(hour_format) + " " + end_cur.strftime(hour_format)
            if verbose:
                print(f"downloading {timeframe} ... ", end="")
            agent = ua.random()
            #print("Custom agent : ", agent)
            # Change of IP address
            old_ip = requests.get('http://icanhazip.com/', proxies={'http': '127.0.0.1:8118'})
            print("Old IP address : ", old_ip.text.strip())
            renew_tor_ip()
            current_ip = requests.get('http://icanhazip.com/', proxies={'http': '127.0.0.1:8118'})
            print("Current IP address : ", current_ip.text.strip())
            pytrends = TrendReq(hl="fr-BE", custom_useragent=agent)
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
        except (ResponseError, ReadTimeout):  # use a delay if an error has been received
            delay = random.randint(5, 60)
            if verbose:
                print(f"Error when downloading. Retrying after sleeping during {delay} sec ...")
            sleep(delay)
    return df_tot


if __name__ == "__main__":
    list_topics = {
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

    for title, mid in list_topics.items():
        collect_historical_interest(mid, title, geo='FR-V')