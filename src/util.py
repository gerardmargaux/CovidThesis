# hold constant and utility functions
import pandas as pd
from typing import List, Tuple
import pycountry
import numpy as np

european_geocodes = {
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
    #'SE': 'Sweden',
    #'SI': 'Slovenia',
    #'SK': 'Slovakia',
}

french_region_and_be = {
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

european_adjacency = [
    ('AT', 'CZ'),
    ('AT', 'DE'),
    ('AT', 'HU'),
    ('AT', 'IT'),
    ('AT', 'SI'),
    ('AT', 'SK'),
    ('BE', 'DE'),
    ('BE', 'FR'),
    ('BE', 'LU'),
    ('BE', 'NL'),
    ('BG', 'GR'),
    ('BG', 'RO'),
    # CY is an island -> no neighbor
    ('CZ', 'DE'),
    ('CZ', 'PL'),
    ('CZ', 'SK'),
    ('DE', 'DK'),
    ('DE', 'FR'),
    ('DE', 'LU'),
    ('DE', 'NL'),
    ('DE', 'PL'),
    ('DK', 'SE'),  # Denmark is considered to be adjacent to Sweden
    ('EE', 'LV'),
    ('ES', 'FR'),
    ('ES', 'PT'),
    ('FI', 'NO'),
    ('FI', 'SE'),
    ('FR', 'IT'),
    ('FR', 'LU'),
    ('HR', 'HU'),
    ('HR', 'SI'),
    ('HU', 'RO'),
    ('HU', 'SI'),
    ('HU', 'SK'),
    # no neighbor for Iceland and for Ireland
    ('IT', 'SI'),
    ('LT', 'LV'),
    ('LT', 'PL'),
    # malta is an island -> no neighbor
    ('NO', 'SE'),
    ('PL', 'SK'),
]

france_region_adjacency = [
    ('FR-A', 'FR-M'),
    ('FR-A', 'FR-I'),
    ('FR-B', 'FR-T'),
    ('FR-B', 'FR-L'),
    ('FR-B', 'FR-N'),
    ('FR-C', 'FR-F'),
    ('FR-C', 'FR-D'),
    ('FR-C', 'FR-V'),
    ('FR-C', 'FR-K'),
    ('FR-C', 'FR-N'),
    ('FR-C', 'FR-L'),
    ('FR-D', 'FR-J'),
    ('FR-D', 'FR-G'),
    ('FR-D', 'FR-I'),
    ('FR-D', 'FR-V'),
    ('FR-D', 'FR-F'),
    ('FR-E', 'FR-P'),
    ('FR-E', 'FR-R'),
    ('FR-F', 'FR-Q'),
    ('FR-F', 'FR-J'),
    ('FR-F', 'FR-L'),
    ('FR-F', 'FR-T'),
    ('FR-F', 'FR-R'),
    ('FR-F', 'FR-P'),
    ('FR-G', 'FR-M'),
    ('FR-G', 'FR-I'),
    ('FR-G', 'FR-J'),
    ('FR-G', 'FR-S'),
    # no adjacent region for FR-H
    ('FR-I', 'FR-M'),
    ('FR-I', 'FR-V'),
    ('FR-J', 'FR-S'),
    ('FR-J', 'FR-Q'),
    ('FR-K', 'FR-V'),
    ('FR-K', 'FR-U'),
    ('FR-K', 'FR-N'),
    ('FR-L', 'FR-N'),
    ('FR-L', 'FR-T'),
    ('FR-O', 'FR-S'),
    ('FR-P', 'FR-Q'),
    ('FR-P', 'FR-R'),
    ('FR-Q', 'FR-S'),
    ('FR-R', 'FR-T'),
    ('FR-U', 'FR-V'),
]

# source: https://en.wikipedia.org/wiki/List_of_European_countries_by_population (UN estimate)
european_population = {
    'AT':  9_006_398,
    'BE': 11_589_623,
    'BG':  6_948_445,
    'CY':  1_195_750,
    'CZ': 10_729_333,
    'DE': 83_783_942,
    'DK':  5_805_607,
    'EE':  1_330_299,
    'ES': 46_811_531,
    'FI':  5_548_480,
    'FR': 65_273_511,
    'GR': 10_391_029,
    'HR':  4_086_308,
    'HU':  9_646_008,
    'IS':    343_008,
    'IE':  4_992_908,
    'IT': 60_461_826,
    'LT':  2_690_259,
    'LU':    635_755,
    'LV':  1_870_386,
    'MT':    514_564,
    'NL': 17_161_189,
    'NO':  5_449_099,
    'PL': 37_830_336,
    'PT': 10_175_378,
    'RO': 19_126_264,
    'SE': 10_147_405,
    'SI':  2_080_044,
    'SK':  5_463_818,
}


def world_adjacency_list(alpha: int = 3) -> List[Tuple[str, str]]:
    """
    :param alpha: iso code to use. Must be in [2, 3]
    :return list of adjacent countries. Neighboring is listed only once, not twice.
    """
    #df = pd.read_csv("https://raw.githubusercontent.com/geodatasource/country-borders/master/GEODATASOURCE-COUNTRY-BORDERS.CSV").dropna()
    df = pd.read_csv("../data/country_borders.csv").dropna()
    adj_list = []
    for idx, row in df.iterrows():
        if row['country_code'] < row['country_border_code']:  # list neighboring once only
            adj_list.append((row['country_code'], row['country_border_code']))
    if alpha == 2:
        return adj_list
    elif alpha == 3:
        return [(alpha2_to_alpha3(a), alpha2_to_alpha3(b)) for a, b in adj_list]
    else:
        raise ValueError("alpha must be 2 or 3")


def alpha2_to_alpha3(code):  # transform a country code from alpha 2 to alpha 3
    return pycountry.countries.get(alpha_2=code).alpha_3


def alpha3_to_alpha2(code):  # transform a country code from alpha 3 to alpha 2
    return pycountry.countries.get(alpha_3=code).alpha_2
