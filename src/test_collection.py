import datetime
import random
from toripchanger import TorIpChanger
from pytrends.request import TrendReq


def random_query():
    geo = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
    'GB': 'Great Britain',
    'GR': 'Greece',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IS': 'Iceland',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
    }
    geo_list = list(geo.keys())  # random.choice does not work on dic
    topics = {
        'Fièvre': '/m/0cjf0',
        'Mal de gorge': '/m/0b76bty',
        'Dyspnée': '/m/01cdt5',
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
    kw = [[code] for code in topics.values()]

    def random_timeframe():
        end_date = datetime.date(year=random.randint(2006, 2020), month=random.randint(1, 12), day=random.randint(1, 28))
        delta = datetime.timedelta(days=random.randint(8, 260))
        beign_date = end_date - delta
        timeframe = f"{beign_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        return timeframe

    while True:
        loc = random.choice(geo_list)
        search = random.choice(kw)
        try:
            current_ip = tor_ip_changer.get_new_ip()
        except:
            pass
        pytrends = TrendReq()
        print(loc)
        print(search)
        pytrends.build_payload(search, cat=0, timeframe=random_timeframe(), geo=loc)
        df = pytrends.interest_over_time()


if __name__ == "__main__":

    tor_ip_changer = TorIpChanger(tor_password='my password', tor_port=9051, local_http_proxy='127.0.0.1:8118')
    random_query()
