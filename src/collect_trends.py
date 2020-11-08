from time import sleep
from pytrends.exceptions import ResponseError
import pandas as pd
import os.path
from datetime import date, datetime, timedelta

from src.request_trends import TrendReq
from my_fake_useragent import UserAgent
import random
import socket
import requests
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller
from requests.exceptions import ReadTimeout

ua = UserAgent()

import requests
import time
from stem import Signal
from stem.control import Controller


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
        collect_historical_interest(mid, title, geo='FR-A')