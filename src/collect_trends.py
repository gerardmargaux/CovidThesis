from __future__ import absolute_import, print_function, unicode_literals
from time import sleep
from pytrends.exceptions import ResponseError
import os.path

from my_fake_useragent import UserAgent
import random
from requests.exceptions import ReadTimeout

from stem import Signal
from stem.control import Controller

import json
from datetime import datetime, timedelta

import pandas as pd
import requests

from pytrends import exceptions

ua = UserAgent()


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


"""def get_current_ip():
    session = requests.session()

    # TO Request URL with SOCKS over TOR
    session.proxies = {}
    session.proxies['http']='socks5h://localhost:9050'
    session.proxies['https']='socks5h://localhost:9050'

    try:
        r = session.get('http://httpbin.org/ip')
    except Exception as e:
        print(e)
    else:
        return r.text"""


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
            #sleep(delay + random.random())
            timeframe = begin_cur.strftime(hour_format) + " " + end_cur.strftime(hour_format)
            if verbose:
                print(f"downloading {timeframe} ... ", end="")
            agent = ua.random()
            print("Custom agent : ", agent)
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