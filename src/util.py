# hold constant and utility functions
import unittest

from typing import List, Tuple, Dict, Iterable, Union
import pycountry
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from copy import deepcopy
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
import networkx as nx
from functools import reduce
import requests
import io
import re


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
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
}

# population above 1m people
european_geocodes_1m = {k: v for k, v in european_geocodes.items() if k not in european_population or european_population[k] > 1_000_000}

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
    'BE': "Belgique",
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

list_topics = {
    'Agueusie': '/m/05sfr2',
    'Allergy': '/m/0fd23',
    'Anosmie': '/m/0m7pl',
    'Coronavirus disease 2019': '/g/11j2cc_qll',
    'COVID 19 testing': '/g/11j8qdq0kc',
    'COVID 19 vaccine': '/g/11j8_9sv06',
    'Cure': '/m/0405g08',
    'Dyspnée': '/m/01cdt5',
    'Fièvre': '/m/0cjf0',
    'Grippe espagnole': '/m/01c751',
    'Mal de gorge': '/m/0b76bty',
    'Paracétamol': '/m/0lbt3',
    'PCR': '/m/05w_j',
    'Respiration': '/m/02gy9_',
    'Respiratory syncytial virus': '/m/02f84_',
    'Severe acute respiratory syndrome coronavirus 2': '/g/11j4xt9hdf',
    'Symptôme': '/m/01b_06',
    'Thermomètre': '/m/07mf1',
    'Toux': '/m/01b_21',
    'Vaccination': '/g/121j1nlf',
    'Virus': '/m/0g9pc',
    'Épidémie': '/m/0hn9s',
}

list_topics_fr = {
    'Agueusie': '/m/05sfr2',
    'Allergy': '/m/0fd23',
    'Anosmie': '/m/0m7pl',
    'Coronavirus disease 2019': '/g/11j2cc_qll',
    'COVID 19 testing': '/g/11j8qdq0kc',
    'COVID 19 vaccine': '/g/11j8_9sv06',
    'Cure': '/m/0405g08',
    'Dyspnée': '/m/01cdt5',
    'Fièvre': '/m/0cjf0',
    'Grippe espagnole': '/m/01c751',
    'Mal de gorge': '/m/0b76bty',
    'Paracétamol': '/m/0lbt3',
    'PCR': '/m/05w_j',
    'Respiration': '/m/02gy9_',
    # 'Respiratory syncytial virus': '/m/02f84_',
    # 'Severe acute respiratory syndrome coronavirus 2': '/g/11j4xt9hdf',
    'Symptôme': '/m/01b_06',
    'Thermomètre': '/m/07mf1',
    'Toux': '/m/01b_21',
    'Vaccination': '/g/121j1nlf',
    'Virus': '/m/0g9pc',
    'Épidémie': '/m/0hn9s',
}

list_topics_eu = {
    'Agueusie': '/m/05sfr2',
    'Allergy': '/m/0fd23',
    'Anosmie': '/m/0m7pl',
    'Coronavirus disease 2019': '/g/11j2cc_qll',
    'COVID 19 testing': '/g/11j8qdq0kc',
    'COVID 19 vaccine': '/g/11j8_9sv06',
    'Cure': '/m/0405g08',
    'Dyspnée': '/m/01cdt5',
    'Fièvre': '/m/0cjf0',
    'Grippe espagnole': '/m/01c751',
    'Mal de gorge': '/m/0b76bty',
    'Paracétamol': '/m/0lbt3',
    'PCR': '/m/05w_j',
    'Respiration': '/m/02gy9_',
    'Respiratory syncytial virus': '/m/02f84_',
    'Severe acute respiratory syndrome coronavirus 2': '/g/11j4xt9hdf',
    'Symptôme': '/m/01b_06',
    'Thermomètre': '/m/07mf1',
    'Toux': '/m/01b_21',
    'Vaccination': '/g/121j1nlf',
    'Virus': '/m/0g9pc',
    'Épidémie': '/m/0hn9s',
}


def datetime_to_str(x: datetime, freq: str):
    mapping = {
        'M': lambda y: y.strftime('%b %Y'),
        'W': lambda y: y.strftime('%Y week %U'),
        'D': lambda y: y.strftime('%Y-%m-%d'),
    }
    return mapping[freq](x)


def log_values(df: pd.DataFrame, columns: list = None, base: int = 10, inf_value=0) -> pd.DataFrame:
    """
    add log values to the dataframe
    :param df: dataframe to change
    :param columns: list of name of the columns that needs to be modified. None= all columns
    :param base: base for the logarithm. Supported: [10]. If not in the list, use logarithm in base e
    :param inf_value: value to give for the inf created by the log. Can be integer or 'drop' (dropping the values)
    :return dataframe with log values for the corresponding columns
    """
    if columns == None:
        columns = df.columns
    new_columns = [f"{name}_log" for name in columns]

    if base == 10:
        df[new_columns] = np.log10(df[columns])
    else:
        df[new_columns] = np.log(df[columns]) / np.log(base)

    if inf_value == 'drop':
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    else:  # inf_value should be an integer
        df = df.replace([np.inf, -np.inf], inf_value)
    return df


def pct_values(df: pd.DataFrame, columns: list = None, add_one: bool = False, threshold=0.5) -> pd.DataFrame:
    """
    add percentage values to the dataframe
    :param df: dataframe to change
    :param columns: list of name of the columns that needs to be modified. None= all columns
    :param add_one: if True, the percentage of difference add a value of 100% to each percentage
    :return dataframe with pct change values for the corresponding columns
    """
    if columns == None:
        columns = df.columns
    new_columns = [f"{name}_pct" for name in columns]
    df[new_columns] = df[columns].pct_change().replace([np.nan, np.inf], 0)
    if add_one:
        df[new_columns] = df[new_columns] + 1
    return df


def get_world_population(pop_file: str, alpha2: bool = True) -> Dict[str, float]:
    """
    :param pop_file: path to the population file, registered as a dict
    :param alpha2: whether to return the dict with the keys being alpha 2 coded or not
    :return dict of population
    """
    pop = json.load(open(pop_file))
    if alpha2:
        return {alpha3_to_alpha2(k): v for k, v in pop.items() if len(k) == 3}
    else:
        return pop


def hospi_french_region_and_be(hospi_france_tot, hospi_france_new, hospi_belgium, department_france, geo,
                               new_hosp=True, tot_hosp=True, new_hosp_in=False, date_begin: str = None):
    """
    Creates the dataframe containing the number of daily hospitalizations in Belgium and in the french regions
    with respect to the date and the localisation (FR and BE)
    :param hospi_france_tot: url/path for the total french hospitalisations csv
    :param hospi_france_new: url/path for the new french hospitalisations csv
    :param hospi_belgium: url/path for the belgian hospitalisations csv
    :param department_france: url/path for the mapping of french department to regions
    :param geo: geocode of the region that should be incuded in the final dict
    :param new_hosp_in: if True, includes the new daily hospitalisations (inwards)
    :param tot_hosp: if True, includes the total hospitalisations
    :return dict of {geocode: hosp_df} where hosp is the hospitalisation dataframe of each geocode
    """
    columns_be = {}  # only for belgium, not for france (the files are handled differently)
    data_columns = []  # final data columns that will be present in the df
    if new_hosp_in:
        columns_be['NEW_IN'] = 'sum'
        data_columns.append("NEW_HOSP_IN")
    if tot_hosp:
        columns_be['TOTAL_IN'] = 'sum'
        data_columns.append("TOT_HOSP")
    if len(columns_be) == 0:
        raise Exception("no hospitalisation column specified")
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    departements = pd.read_csv(department_france)

    # French data: total hospitalisation
    if tot_hosp or new_hosp:
        hospitalisations = pd.read_csv(hospi_france_tot, sep=";", parse_dates=['jour'], date_parser=date_parser)
        hospitalisations = hospitalisations[hospitalisations['sexe'] == 0]  # sex=0: men and women
        data_fr_tot = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(
            ["regionTrends", "jour"], as_index=False).agg({"hosp": "sum"})

    # French data: new hospitalisation
    if new_hosp_in:
        hospitalisations = pd.read_csv(hospi_france_new, sep=";", parse_dates=['jour'], date_parser=date_parser)
        data_fr_new = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(
            ["regionTrends", "jour"], as_index=False).agg({"incid_hosp": "sum"})

    # merge the french data
    common_columns = ["regionTrends", "jour"]
    if (tot_hosp or new_hosp) and new_hosp_in:
        data_fr = data_fr_tot.merge(data_fr_new, how='outer', left_on=common_columns, right_on=common_columns).fillna(0)
    elif tot_hosp or new_hosp:
        data_fr = data_fr_tot
    elif new_hosp_in:
        data_fr = data_fr_new
    data_fr = data_fr.rename(
        columns={"jour": "DATE", "regionTrends": "LOC", "hosp": "TOT_HOSP", "incid_hosp": "NEW_HOSP_IN"})

    # Belgian data
    data_be = pd.read_csv(hospi_belgium, parse_dates=['DATE'], date_parser=date_parser).groupby(
        ["DATE"], as_index=False).agg(columns_be).rename(
        columns={"TOTAL_IN": "TOT_HOSP", "NEW_IN": "NEW_HOSP_IN"})
    data_be["LOC"] = "BE"

    # Full data
    full_data = data_fr.append(data_be).set_index(["LOC", "DATE"])

    # find smallest date for each loc and highest common date
    smallest = {}
    highest = {}
    for loc, date_current in full_data.index:
        if loc not in smallest or smallest[loc] > date_current:
            smallest[loc] = date_current
        if loc not in highest or highest[loc] < date_current:
            highest[loc] = date_current

    highest_date = min(highest.values())
    if date_begin is None:
        date_begin = '2020-02-01'
    base_date = datetime.strptime(date_begin, "%Y-%m-%d").date()

    # Add "fake" data (zeroes before the beginning of the crisis) for each loc
    toadd = []
    add_entry = [0 for i in range(len(data_columns))]  # each missing entry consist of zero for each data col
    for loc, sm in smallest.items():
        end = sm.date()
        cur = base_date
        while cur != end:
            toadd.append([cur, loc, *add_entry])
            cur += timedelta(days=1)

    full_data = pd.DataFrame(toadd, columns=["DATE", "LOC", *data_columns]).append(full_data.reset_index()).set_index(
        ["LOC", "DATE"])
    data_dic = {}

    for k, v in geo.items():
        data_dic[k] = full_data.iloc[(full_data.index.get_level_values('LOC') == k) &
                                     (full_data.index.get_level_values('DATE') <= highest_date)]
        if new_hosp:
            data_dic[k]['NEW_HOSP'] = data_dic[k]['TOT_HOSP'].diff()
            data_dic[k].at[data_dic[k].index.min(), 'NEW_HOSP'] = 0
    return data_dic


def hospi_world(hospi_file: str, geo: Dict[str, str], renaming: Dict[str, str], date_begin: str,
                tot_hosp=True, new_hosp=False, tot_icu=False, new_icu=False) -> Dict[str, pd.DataFrame]:
    """
    Creates the dataframe containing the number of daily hospitalisations in Europe and
    update the geocodes given in order to remove regions without data
    :param hospi_file: url/path for the hospitalisations csv
    :param geo: geocode of the countries that should be incuded in the final dict. The dict is updated if a
        region does not have data
    :param renaming: renaming to use for the countries
    :param date_begin: date of beginning (format YYYY-MM-DD), 0 will be added from this date until the first date where
        data can be found
    :param tot_hosp: whether or not to give the total hosp in the final df
    :param new_hosp: whether or not to give the new hosp in the final df
    :param tot_icu: whether or not to give the total icu in the final df
    :param new_icu: whether or not to give the new icu in the final df
    :return dict of {geocode: hosp_df} where hosp is the hospitalisation dataframe of each geocode
    """
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    full_data = pd.read_csv(hospi_file, parse_dates=['date'], date_parser=date_parser).rename(
        columns={'iso_code': 'LOC', 'date': 'DATE', 'icu_patients': 'TOT_ICU', 'hosp_patients': 'TOT_HOSP'})
    # transform iso code from alpha 3 to alpha 2
    code_mapping = {}
    for code in full_data['LOC'].unique():
        if len(code) == 3:
            code_mapping[code] = alpha3_to_alpha2(code)

    full_data = full_data.replace({**renaming, **code_mapping})
    full_data = full_data.set_index(["LOC", "DATE"])

    data_columns = []
    if tot_icu:
        data_columns.append('TOT_ICU')
    if tot_hosp:
        data_columns.append('TOT_HOSP')
    full_data = full_data[data_columns]

    if new_hosp:
        data_columns.append('NEW_HOSP')
    if new_icu:
        data_columns.append('NEW_ICU')
    add_entry = [0 for _ in range(len(data_columns))]
    data_dic = {}
    base_date = datetime.strptime(date_begin, "%Y-%m-%d").date()
    country_to_remove = []

    for loc in geo:
        df = full_data.iloc[full_data.index.get_level_values('LOC') == loc]
        # reindex in case missing values without NaN appear
        min_date = df.index.get_level_values('DATE').min()
        max_date = df.index.get_level_values('DATE').max()
        reindexing = pd.MultiIndex.from_product([[loc], pd.date_range(min_date, max_date)], names=['LOC', 'DATE'])
        df = df.reindex(reindexing, fill_value=np.nan)
        df = df.interpolate(limit_area='inside').dropna()
        # remove NaN entries
        df = df.dropna()
        # the dataframe might not have any entry at this point -> remove it if it the case
        if df.empty:
            print(f"region {loc} does not have any entry, removing it")
            country_to_remove.append(loc)
            continue
        # add zeros at the beginning if no data is found
        smallest_date = df.index.get_level_values('DATE').min()

        to_add = []
        end = smallest_date
        cur = base_date
        if cur <= end:
            while cur != end:
                to_add.append([cur, loc, *add_entry])
                cur += timedelta(days=1)
            df = pd.DataFrame(to_add, columns=["DATE", "LOC", *data_columns]).append(df.reset_index()).set_index(
                ["LOC", "DATE"])
        else:  # drop data if it is too early
            begin = datetime.fromordinal(base_date.toordinal())
            df = df.iloc[df.index.get_level_values('DATE') >= begin]
        # add the relevant new columns
        if new_icu:
            df['NEW_ICU'] = df['TOT_ICU'].diff()
            df.at[df.index.min(), 'NEW_ICU'] = 0
        if new_hosp:
            df['NEW_HOSP'] = df['TOT_HOSP'].diff()
            df.at[df.index.min(), 'NEW_HOSP'] = 0
        data_dic[loc] = df
    for loc in country_to_remove:
        del geo[loc]
    return data_dic


def add_transformations(dict_df: Dict[str, pd.DataFrame], list_transformations: List[str]) -> Dict[str, pd.DataFrame]:
    """
    add a list of transformations to an existing dataframe, leading to newly created columns
    :param dict_df: dict of loc: dataframe of data
    :param list_transformations: list of transformations that needs to be performed
        each transformation must be named as "feature_{transformation}"
        accepted transformations are
            - pct: for percentage change of the feature
            - log: for log change of the feature
        invalid format are ignored
    :return: dict of transformed dataframe, with new features added
    """
    accepted_transformation = {
        'pct': pct_values,
        'log': log_values,
    }
    pattern_transformation = ''
    for transformation in accepted_transformation:
        pattern_transformation += '|' + transformation
    pattern_transformation = pattern_transformation[1:]
    for feat_transformation in list_transformations:
        search_obj = re.search(f'(.*)_({pattern_transformation})', feat_transformation)
        if search_obj is None:
            continue
        try:
            feature = search_obj.group(1)
            transformation = search_obj.group(2)
            for loc, df in dict_df.items():
                vals = accepted_transformation[transformation](df[[feature]])[feat_transformation]
                df[feat_transformation] = vals
        except:
            continue
    return dict_df


def create_df_trends(url_trends: str, list_topics: Dict[str, str], geo: Dict[str, str],
                     diff_trends: bool = False) -> Dict[str, pd.DataFrame]:
    """
    return dic of {geo: df} for the trends
    :param url_trends: path to the trends data folder
    :param list_topics: dict of topic title: topic code for each google trends
    :param geo: dict of geo localisations to use
    :param diff_trends: whether to add relative augmentation of trends (.diff)
    """
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    renaming = {v: k for k, v in list_topics.items()}  # change topic_mid to topic_title in the dataframe
    if len(renaming) == 0:
        return {k: pd.DataFrame() for k in geo}
    result = {}
    for k, v in geo.items():
        all_trends = []
        for term in list_topics.keys():
            path = f"{url_trends}{k}-{term}.csv"
            if url_trends[:4] == "http":
                encoded_path = requests.get(path).content
                df_trends = pd.read_csv(io.StringIO(encoded_path.decode("utf-8")), parse_dates=['date'],
                                        date_parser=date_parser).rename(columns={"date": "DATE"})
            else:
                df_trends = pd.read_csv(path, parse_dates=['date'], date_parser=date_parser).rename(
                    columns={"date": "DATE"})
            df_trends['LOC'] = k
            df_trends.rename(columns=renaming, inplace=True)
            df_trends.set_index(['LOC', 'DATE'], inplace=True)
            if diff_trends:
                new_term = f'NEW_{term}'
                new_trends = df_trends.diff().rename(columns={term: new_term})
                new_trends.iloc[0] = 0
                df_trends[new_term] = new_trends
            all_trends.append(df_trends)
        result[k] = pd.concat(all_trends, axis=1)
    return result


def world_adjacency_list(alpha: int = 3) -> List[Tuple[str, str]]:
    """
    :param alpha: iso code to use. Must be in [2, 3]
    :return list of adjacent countries. Neighboring is listed only once, not twice.
    """
    # df = pd.read_csv("https://raw.githubusercontent.com/geodatasource/country-borders/master/GEODATASOURCE-COUNTRY-BORDERS.CSV").dropna()
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


def add_lag(df, lag, dropna=True):
    """
    add lagged columns to a dataframe
    :param df: dataframe to modify
    :param lag: lag to add. Values can be negative (old values) or positive (forecast values)
        if positive, lag values from the future are added, excluding the ones from today
        otherwise, lag-1 values from the past are added, including the ones from today
    :param dropna: if True, drop the NaN columns created
    """
    if lag < 0:  # add values from the past
        lag_range = range(lag + 1, 0, 1)
    else:  # add values from the future
        lag_range = range(1, lag + 1, 1)
    columns = []

    init_names = df.columns
    for i in lag_range:
        renaming = {col: f"{col}(t{i:+d})" for col in init_names}  # name of the lagged columns
        columns.append(df.shift(-i).rename(columns=renaming))

    if lag < 0:
        columns.append(df)  # include the original data if lag < 0
    df = pd.concat(columns, axis=1)
    return df.dropna() if dropna else df


def region_merge_iterator(init_regions: List[str], nb_merge: int, adj: List[Tuple[str, str]] = None):
    """
    yield list of regions, supposed to be used to form augmented regions
    :param init_regions: list of regions to merge
    :param nb_merge: number of regions that can be used (at most) to create an augmented region. Must be >=2
    :param adj: list of adjacency to use. If None, all regions will be mixed, even unadjacent. Otherwise
        use the list of adjacent region to augment the data
    :return: yield list of regions to merge
    """
    if adj:
        G = nx.Graph(adj)
    for merge in range(2, nb_merge + 1):
        for elem in itertools.combinations(init_regions, merge):
            if adj:  # check if the region candidate can be formed
                connected = [False for _ in range(len(elem))]
                for i, node_a in enumerate(elem):
                    for j, node_b in enumerate(elem[i + 1:]):
                        if G.has_edge(node_a, node_b):
                            connected[i] = True
                            connected[i + j + 1] = True
                if np.all(connected):
                    yield elem
            else:  # unadjacent regions can be given
                yield elem


class DataGenerator:

    def __init__(self,
                 df: Dict[str, pd.DataFrame],
                 n_samples: int,
                 n_forecast: int,
                 target: str,
                 scaler_generator: callable,
                 scaler_type: str = "batch",
                 data_columns: List[str] = None,
                 no_scaling: List[str] = None,
                 cumsum: bool = False,
                 predict_one: bool = False,
                 augment_merge: int = 1,
                 augment_adjacency: List[Tuple[str, str]] = None,
                 augment_population: Dict[str, Union[int, float]] = None,
                 augment_feature_pop: List[str] = None,
                 no_lag: bool = False):
        """
        initialize a data generator. Takes a dict of {loc: dataframe} and use it to yield values suitable for training
        the data generator can augment the data by mixing regions together
        the values are padded so that each region contains the same number of datapoints. Only the right padding information is stored

        :param df: dataframe of values to use in order to generate X and Y. Must be double indexed by loc and date
        :param n_samples: number of timesteps in X
        :param n_forecast: number of timesteps in Y. If == 0, no target is set
        :param target: target to predict, will become the Y set. If == '', no target is set
        :param scaler_generator: generator of scaler to use
        :param scaler_type: one of "batch", "window", "whole"
        :param no_scaling: list of features that must not be scaled
        :param cumsum: if True, accumulates Y using cumsum
        :param predict_one: if True, the target is Y at time n_forecast. Otherwise, the target is Y in [t+1 ... t+n_forecast]
        :param augment_merge: number of regions to merge in order to augment the data. If <=1, no data augmentation is performed
        :param augment_adjacency: use the list of adjacent region to augment the data. If None, all regions
            will be mixed, even unadjacent
        :param augment_population: population of each region. Must not be None if augment_feature_pop contains values
        :param augment_feature_pop: list of features that should be weighted according to the population
        :param no_lag: if True, the dataframe will be considered as having the right format for training and add_lag
            will not be called on it
        """
        self.n_samples = n_samples
        self.n_forecast = n_forecast
        self.target = target
        self.cumsum = cumsum
        # transform the dict of dataframe into shape [samples, time_steps, features]
        # add lag to the data to be able to reshape correctly
        df = deepcopy(df)
        dummy_df = next(iter(df.values()))
        self.nb_loc = len(df)
        init_columns = list(dummy_df.columns)
        if data_columns is None:
            self.n_features = dummy_df.shape[1]
            data_columns = init_columns  # all columns are considered to be data columns
        else:
            self.n_features = len(data_columns)
        if target in data_columns:
            self.target_idx = data_columns.index(target)
            # TODO use option with cumsum
            if not cumsum:
                self.target_in_x = True
            else:
                self.target_in_x = False
        else:
            self.target_idx = None  # the target is not in the data columns, no need to specify it
            self.target_in_x = False

        # handle data generator without target
        self.no_target = target == '' or n_forecast == 0  # no target specified

        # pad the values: add 0 at the beginning and at the end
        smallest_dates = {}
        highest_dates = {}
        for k, v in df.items():  # get the dates covered on each loc
            dates = v.index.get_level_values('DATE')
            smallest_dates[k] = dates.min()
            highest_dates[k] = dates.max()
        min_date = min(smallest_dates.values())
        max_date = max(highest_dates.values())

        # add zeros (missing data) at the beginning and at the end, so that each df has the same number of values
        self.padded_idx = {}
        for k in df:
            if smallest_dates[k] > min_date:  # missing data at the beginning
                date_range = pd.date_range(min_date, smallest_dates[k] - timedelta(days=1))
                # loc_padded_idx = np.array(range(len(date_range)))
                nb_point = len(date_range)
                zeros = np.zeros(nb_point)
                pad_before = pd.DataFrame({**{'DATE': date_range, 'LOC': [k for _ in range(nb_point)]},
                                           **{col: zeros for col in init_columns}}).set_index(["LOC", "DATE"])
                df[k] = pad_before.append(df[k])
            else:
                pass
                # loc_padded_idx = np.array([])
            if highest_dates[k] < max_date:  # missing data at the end
                date_range = pd.date_range(highest_dates[k] + timedelta(days=1), max_date)
                #loc_padded_idx = np.append(loc_padded_idx, range(len(df[k]), len(df[k]) + len(date_range)))
                loc_padded_idx = np.arange(len(df[k]), len(df[k]) + len(date_range))
                nb_point = len(date_range)
                zeros = np.zeros(nb_point)
                pad_after = pd.DataFrame({**{'DATE': date_range, 'LOC': [k for _ in range(nb_point)]},
                                          **{col: zeros for col in init_columns}}).set_index(["LOC", "DATE"])
                df[k] = df[k].append(pad_after)
            else:
                loc_padded_idx = np.array([])
            self.padded_idx[k] = loc_padded_idx

        # augment the data
        self.loc_init = {k: k for k in df}
        self.loc_augmented = {}
        if augment_merge > 1:
            if augment_feature_pop is None:
                augment_feature_no_pop = init_columns
            else:
                augment_feature_no_pop = [col for col in init_columns if col not in augment_feature_pop]
                # filter to only take the columns already present in the dataframe
                augment_feature_pop = [i for i in augment_feature_pop if i in init_columns]
            for region_list in region_merge_iterator([loc for loc in df], augment_merge, augment_adjacency):
                region_code = '-'.join(sorted(set(region_list)))
                df[region_code] = sum(
                    [df[k][augment_feature_no_pop].reset_index().drop(columns=['LOC']).set_index('DATE') for k in
                     region_list])
                if augment_feature_pop:
                    sum_pop = sum([augment_population[k] for k in region_list])
                    df[region_code][augment_feature_pop] = sum(
                        [df[k][augment_feature_pop].reset_index().drop(columns=["LOC"]).
                        set_index('DATE') * augment_population[k] for k in region_list]) / sum_pop

                df[region_code]['LOC'] = region_code
                df[region_code] = df[region_code].reset_index().set_index(['LOC', 'DATE'])
                df[region_code] = df[region_code][init_columns]
                self.loc_augmented[region_code] = "Augmented region"

                if len(region_list) == 2:
                    self.padded_idx[region_code] = np.union1d(self.padded_idx[region_list[0]],
                                                              self.padded_idx[region_list[1]])
                else:
                    self.padded_idx[region_code] = reduce(np.union1d, ([self.padded_idx[k] for k in region_list]))
        # add data transformation
        add_transformations(df, data_columns)

        self.loc_all = {**self.loc_init, **self.loc_augmented}
        self.df_init = df  # contains the augmented data with padding and without lagged values
        self.padded_idx_init = deepcopy(self.padded_idx)  # padded indexes before the add lag
        # get the target and data columns
        self.predict_one = predict_one
        if predict_one:
            target_columns = [f"{target}(t+{n_forecast})"]
        else:
            target_columns = [f"{target}(t+{t})" for t in range(1, n_forecast + 1)]
        self.target_columns = target_columns
        self.data_columns_t0 = deepcopy(data_columns)
        # name of x columns across time
        self.data_columns = [f"{col}(t{t:+d})" for t in range(-n_samples + 1, 0) for col in
                             data_columns] + self.data_columns_t0

        # add lagged values
        if no_lag:  # the dataframe has already the right format
            self.df = df
            self.date_range = pd.date_range(min_date, max_date).to_pydatetime()
            days_removed = 0  # no day as been removed
        else:  # the dataframe must be constructed across time
            if self.no_target:  # no target specified
                self.df = {k: add_lag(v, - n_samples) for k, v in df.items()}
                self.date_range = pd.date_range(min_date + timedelta(days=(n_samples - 1)),
                                                max_date).to_pydatetime()
                days_removed = n_samples - 1
            else:  # a target exist
                self.df = {k: add_lag(v, - n_samples).join(add_lag(v[[target]], n_forecast),
                                                           how='inner') for k, v in df.items()}
                self.date_range = pd.date_range(min_date + timedelta(days=(n_samples - 1)),
                                                max_date - timedelta(days=n_forecast)).to_pydatetime()
                days_removed = n_forecast + n_samples - 1
        self.padded_idx = {k: (v - days_removed).astype(int) for k, v in self.padded_idx.items()}

        self.scaler_type = scaler_type  # can be "batch", "window", "whole"
        if no_scaling is None or target not in no_scaling:
            self.target_unscaled = False
        else:
            self.target_unscaled = True
        # unscaled columns names accross horizon
        if no_scaling is None:
            self.to_scale = list(range(self.n_features))
        else:
            no_scaling = [dummy_df.columns.get_loc(f"{name}(t-{n_samples})") for name in no_scaling]
            self.to_scale = [i for i in range(self.n_features) if i not in no_scaling]

        # construct the X and Y tensors
        self.X = []
        self.Y = []
        self.idx = {}
        idx = 0
        for i, (loc, val) in enumerate(self.df.items()):
            if not self.no_target:  # a target is specified
                y = val[target_columns].values
                if cumsum:
                    y = np.cumsum(y, axis=1)
                self.Y.append(y)
            x = val[self.data_columns].values
            x = x.reshape((len(x), n_samples, self.n_features))
            self.X.append(x)
            self.batch_size = len(x)
            self.idx[loc] = np.array(range(idx, idx + self.batch_size))
            idx += self.batch_size
            if i == 0:
                self.relative_idx = np.array(range(0, self.batch_size))
        self.X = np.concatenate(self.X)
        if self.no_target:
            self.Y = None
        else:
            self.Y = np.concatenate(self.Y)

        # self.X = np.concatenate([val.values.reshape((val.shape[0], n_samples, self.n_features))[:-n_forecast] for val in self.df.values()], axis=0)
        # self.Y = np.concatenate([val[target_columns].iloc[n_forecast:] for val in self.df.values()], axis=0)
        # self.Y = np.concatenate([val[target_columns].iloc[n_forecast:].cumsum(axis=1) for val in self.df.values()], axis=0)

        self.scaler_generator = None  # initialized by the set_scaler method
        self.set_scaler(scaler_generator)

    def set_scaler(self, scaler_generator: callable):
        """
        set the scaler used by the datagenerator
        :param scaler_generator: function that can be called to give the scaler to use
        """
        self.scaler_generator = scaler_generator
        # set the scaler
        # if window: Dict[str, Dict[int, [Dict[int, scaler]]]]: {loc: {feature_idx: {idx: scaler}}}
        # else: Dict[str, Dict[int, scaler]]: {loc: {feature_idx: scaler}}
        if self.scaler_type == "window":  # relative index
            self.scaler_x = {
                loc: {feature: {i: self.scaler_generator() for i in self.relative_idx} for feature in self.to_scale} for
                loc in self.idx}
            if self.no_target:
                self.scaler_y = None
            else:
                self.scaler_y = {loc: {i: self.scaler_generator() for i in self.relative_idx} for loc in self.idx}
        else:
            self.scaler_x = {loc: {feature: self.scaler_generator() for feature in self.to_scale} for loc in self.idx}
            if self.no_target:
                self.scaler_y = None
            else:
                self.scaler_y = {loc: self.scaler_generator() for loc in self.idx}

    def set_scaler_values_x(self, scaler_x: Dict):
        """
        copies the values of a scaler dict to this scaler
        """
        self.scaler_x = deepcopy(scaler_x)

    def set_loc_init(self, loc: Dict[str, str]):
        """
        set the initial loc and change the other loc to the augmented status
        :param loc: dict of localisation to consider as unaugmented localisations
        """
        self.loc_init = deepcopy(loc)
        self.loc_augmented = {k: v for k, v in self.loc_all.items() if k not in self.loc_init}

    def set_padded_idx(self, padded_idx):
        pass

    def get_x_dates(self, idx: Iterable = None):
        """
        gives the dates of the x values
        :param idx: index of dates of the x values to provide. If None, provide all possible dates
        :return array of dates, in one dimension
        """
        return self.date_range.to_pydatetime() if idx is None else self.date_range.to_pydatetime()[idx]

    def get_y_dates(self, idx: Iterable = None):
        """
        gives the dates of the y values
        :param idx: index of dates of the y values to provide. If None, provide all possible dates
        :return array of dates. If not self.predict_one, the array is 2d: [idx, n_forecast] else the array is 1d
        """
        if self.predict_one:
            dates = np.array([(self.date_range + timedelta(days=self.n_forecast)).to_pydatetime()])
        else:
            dates = np.column_stack(
                [(self.date_range + timedelta(days=i)).to_pydatetime() for i in range(1, self.n_forecast + 1)])
        return dates if idx is None else dates[idx]

    def get_x(self, idx: Iterable = None, geo: Dict[str, str] = None, scaled=True,
              use_previous_scaler: bool = False) -> np.array:
        """
        gives a X tensor, used to predict Y
        :param idx: index of the x values to provide. If None, provide the whole x values. Must be specified if
            scaler_type == 'batch'
        :param geo: localisations asked. If None, provide all loc
        :param scaled: if True, scale the data. Otherwhise, gives unscaled data
        :param use_previous_scaler: if True, use the scalers that were fit previously instead of new ones
        :return tensor of X values on the asked geo localisations and asked indexes.
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        if idx is not None:
            idx = np.array(idx)
        else:
            idx = self.relative_idx
        X = []
        for loc in geo:
            val = self.X[self.idx[loc], :]
            if not scaled:
                val = val[idx, :, :]
            else:  # need to scale x
                if self.scaler_type == "batch" or self.scaler_type == "whole":
                    if self.scaler_type == "batch":
                        val = val[idx, :, :]
                    # transform each feature
                    for feature_idx in self.to_scale:
                        if not use_previous_scaler:
                            if self.target_in_x and feature_idx == self.target_idx:
                                # need to add the values of y as well for the scaling
                                y_val = self.Y[self.idx[loc], :]
                                if self.scaler_type == "whole":
                                    y_val = y_val[-1, :]
                                else:
                                    y_val = y_val[idx[-1], :]
                            else:
                                y_val = []
                            old = val[:, -1, feature_idx].reshape(-1)  # get the values at t=0 on each window
                            new = val[0, :-1, feature_idx].reshape(-1)  # add the oldest values in the first window
                            new = np.append(new, y_val)
                            self.scaler_x[loc][feature_idx].fit(np.append(old, new).reshape((-1, 1)))  # fit the scaler
                        for t in range(self.n_samples):  # apply the transformation on the feature across time
                            val[:, t, feature_idx] = self.scaler_x[loc][feature_idx].transform(
                                val[:, t, feature_idx].reshape((-1, 1))).reshape((1, -1))
                    if self.scaler_type == "whole":
                        val = val[idx, :, :]
                elif self.scaler_type == "window":
                    if idx is None:
                        iterator = list(range(len(val)))
                    else:
                        iterator = idx
                    for feature_idx in self.to_scale:  # transform each feature
                        for i in iterator:
                            if use_previous_scaler:
                                val[i, :, feature_idx] = self.scaler_x[loc][feature_idx][i].transform(
                                    val[i, :, feature_idx].reshape((-1, 1))).reshape((1, -1))
                            else:
                                val[i, :, feature_idx] = self.scaler_x[loc][feature_idx][i].fit_transform(
                                    val[i, :, feature_idx].reshape((-1, 1))).reshape((1, -1))
                    val = val[idx, :, :]
            X.append(val)
        return np.concatenate(X)

    def get_y(self, idx: Iterable[int] = None, geo: Dict[str, str] = None, scaled: bool = True,
              use_previous_scaler: bool = False, repeated_values: bool = False) -> np.array:
        """
        gives a Y tensor, that should be predicted based on X
        :param idx: relative indexes of the y values to provide. If None, provide the whole y values
        :param geo: localisations asked. If None, provide all loc
        :param scaled: if True, scale the data. Otherwhise, gives unscaled data
        :param use_previous_scaler: if True, use the scalers that were fit previously instead of new ones
        :param repeated_values: should be False if for a loc y[i, t] == y[i+1, t-1] (This is the default behavior when
            no_lag == True in the constructor). If True, the scaler will be fit on the whole Y matrix
        :return tensor of y values on the asked geo localisations and asked indexes.
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        if idx is not None:
            idx = np.array(idx)
        else:
            idx = self.relative_idx
        Y = []
        for loc in geo:
            val = self.Y[self.idx[loc], :]
            if not scaled or self.target_unscaled:
                val = val[idx, :]
            else:  # need to scale y
                if self.scaler_type == "batch" or self.scaler_type == "whole":
                    if self.scaler_type == "batch":
                        val = val[idx, :]
                    # transform each feature
                    if self.predict_one:
                        if use_previous_scaler:
                            val = self.scaler_y[loc].transform(val)
                        else:
                            val = self.scaler_y[loc].fit_transform(val)
                    else:
                        if not use_previous_scaler:
                            if repeated_values or self.cumsum:
                                self.scaler_y[loc].fit(val.reshape((-1, 1)))  # fit the scaler
                            else:
                                if self.target_in_x:
                                    # need to add the values stored in x
                                    x_val = self.X[self.idx[loc], :, self.target_idx]
                                    if self.scaler_type == "whole":
                                        x_val = x_val[0, :]
                                    else:
                                        x_val = x_val[idx[0], :]
                                else:
                                    x_val = []
                                old = val[:, 0]  # get the values at t+1
                                old = np.append(old, x_val)
                                new = val[-1, 1:]  # add the most recent values at t+2 ... t+n_forecast
                                self.scaler_y[loc].fit(np.append(old, new).reshape((-1, 1)))  # fit the scaler
                        for t in range(val.shape[
                                           1]):  # apply the transformation on the target across time  # TODO optimize in one go
                            val[:, t] = self.scaler_y[loc].transform(val[:, t].reshape((-1, 1))).reshape((1, -1))
                    if self.scaler_type == "whole" and idx is not None:
                        val = val[idx, :]
                elif self.scaler_type == "window":
                    if idx is None:
                        iterator = range(len(val))
                    else:
                        iterator = idx
                    for i in iterator:
                        if use_previous_scaler:
                            val[i, :] = self.scaler_y[loc][i].transform(val[i, :].reshape((-1, 1))).reshape((1, -1))
                        else:
                            val[i, :] = self.scaler_y[loc][i].fit_transform(val[i, :].reshape((-1, 1))).reshape((1, -1))
                    val = val[idx, :]
            Y.append(val)
        return np.concatenate(Y)

    def inverse_transform_y(self, unscaled: np.array, geo: Union[str, Dict[str, str]] = None, idx: np.array = None,
                            return_type: str = 'array', inverse_transform: bool = True) -> Union[
        np.array, Dict[str, pd.DataFrame], Dict[str, np.array]]:
        """
        inverse transform the values provided, in order to get unscaled data
        uses the last scalers used in the get_y method
        :param unscaled: values to scale
        :param geo: localisation to scale. If None, all localisations are used. Must be in the order of Y:
            the first loc corresponds to the idx first entries in Y and so on
        :param idx: relative indexes of the scaling. Only relevant if the scaling type is batch
        :param return_type: can be one of "array", "dict_array" or "dict_df"
            - array: return a 2D np.array of values
            - dict_array: return a dict of {loc: np.array}
            - dict_df: return a dict of {loc: pd.DataFrame}
        :param inverse_transform: don't use any inverse transform
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        elif isinstance(geo, str):
            geo = {geo: self.idx[geo]}
        if idx is None or self.scaler_type != "batch":  # the scaler_type must be 'whole' or 'window'
            idx = self.relative_idx
            idx_dates = np.array(idx)
        else:
            idx_dates = np.array(idx)
            idx = np.array(range(len(idx)))

        val = np.zeros(unscaled.shape)
        return_df = {}
        if self.scaler_type == "batch" or self.scaler_type == "whole":
            offset = 0  # current offset in the Y tensor
            batch_size = len(idx)
            for loc in geo:
                loc_idx = idx + offset
                init_shape = unscaled[loc_idx, :].shape
                if inverse_transform:
                    val[loc_idx, :] = self.scaler_y[loc].inverse_transform(unscaled[loc_idx, :].reshape((-1, 1))).reshape(
                        init_shape)
                else:
                    val[loc_idx, :] = unscaled[loc_idx, :]
                offset += batch_size  # increment the offset to get the values from the next batch
                if return_type == 'dict_df':
                    dates_used = self.date_range[idx_dates]
                    multi_index = pd.MultiIndex.from_product([[loc], dates_used], names=['LOC', 'DATE'])
                    return_df[loc] = pd.DataFrame(val[loc_idx, :], columns=self.target_columns).set_index(
                        multi_index)
                elif return_type == 'dict_array':
                    return_df[loc] = val[loc_idx, :]
        elif self.scaler_type == "window":
            offset = 0  # current offset in the Y tensor
            batch_size = len(idx)
            if inverse_transform:
                for loc in geo:
                    for j, i in enumerate(idx):  # TODO implement inverse transform for window and corresponding return type
                        val[i + offset, :] = self.scaler_y[loc][i].inverse_transform(
                            unscaled[i + offset, :].reshape((-1, 1))).reshape((1, -1))
                    offset += batch_size  # increment the offset to get the values from the next batch
        return val if return_type == 'array' else return_df

    def loc_to_idx(self, loc):  # return absolute idx
        return self.idx[loc]

    def remove_padded_y(self, val: np.array, geo: Union[str, Dict[str, str]] = None, idx: np.array = None,
                        return_type: str = 'array') -> Union[np.array, Dict[str, pd.DataFrame], Dict[str, np.array]]:
        """
        remove the padded values of y and gives the result as a numpy array, or a dict of dataframe or numpy array
        :param val: values to scale
        :param geo: localisation to scale. If None, all localisations are used. Must be in the order of Y:
            the first loc corresponds to the idx first entries in Y and so on
        :param idx: relative indexes of the values. Only relevant if the scaling type is batch
        :param return_type: can be one of "array", "dict_array" or "dict_df"
            - array: return a 2D np.array of values
            - dict_array: return a dict of {loc: np.array}
            - dict_df: return a dict of {loc: pd.DataFrame}
        :return unpadded values of y. The type of return depends of the return_type parameter
        """
        implemented = ['array', 'dict_array', 'dict_df']
        if return_type not in implemented:
            raise Exception(f"return type should be one of {implemented}")

        if geo is None:
            geo = self.idx  # only the keys are needed
        elif isinstance(geo, dict):
            pass
        else:
            geo = {geo: self.idx[geo]}
        if idx is None:
            idx = self.relative_idx
        else:
            idx = np.array(idx)

        # remove the values where the data was padded
        filtered_values = {}
        batch_size = len(idx)
        offset = 0
        for loc in geo:
            # remove the indexes where the data was padded
            unpadded_idx = np.setdiff1d(idx, self.padded_idx[loc])
            dates_used = self.date_range[unpadded_idx]
            unpadded_idx = unpadded_idx + offset - idx[0]  # add offset inside matrix and remove first values if absent
            if return_type == 'dict_df':
                multi_index = pd.MultiIndex.from_product([[loc], dates_used], names=['LOC', 'DATE'])
                filtered_values[loc] = pd.DataFrame(val[unpadded_idx], columns=self.target_columns).set_index(
                    multi_index)
            else:
                filtered_values[loc] = val[unpadded_idx]
            offset += batch_size

        if return_type == 'dict_array' or return_type == 'dict_df':
            return filtered_values
        else:
            return np.concatenate([val for val in filtered_values.values()])

    def get_df_init(self) -> Dict[str, pd.DataFrame]:
        """
        return the initial dataframe that was used to construct the data, removing the padded values
        augmented values are included
        """
        init_df = {}
        for loc, df in self.df_init.items():
            unpadded_idx = np.setdiff1d(range(len(df)), self.padded_idx_init[loc])
            init_df[loc] = df.iloc[unpadded_idx]
        return init_df

    def __str__(self):
        """
        contains informations about
            - n_samples, n_forecast
            - data columns
            - target
            - scaling done
            - number of init and augmented regions, as well as their name
        """
        info = f'n_samples = {self.n_samples}, n_forecast = {self.n_forecast}\n'
        info += f'data = {self.data_columns_t0}\n'
        info += f'target = {self.target}\n'
        info += f'scaling = {self.scaler_generator}, scaling type = {self.scaler_type}\n'
        info += f'nb init regions = {len(self.loc_init)}, nb augmented regions = {len(self.loc_augmented)}\n'
        list_regions = [loc for loc in self.df]
        info += f'regions = {list_regions}'
        return info

    def time_idx(self, freq='M', format_date=False, boundary='inner') -> List[Tuple[np.array, Union[datetime, str]]]:
        """
        give the indexes corresponding to time interval

        :param freq: frequency for the time interval. supported:
            - 'M': monthly data
            - 'W': weekly data
            - 'D': daily data
        :param format_date: If True, transform the datetime into str, based on the frequency
        :param boundary: tell how to proceed for boundary: dates with values overlapping on multiple interval. supported:
            - 'inner': indices are split on dates where the n_forecast targets are in the next interval
            - 'outer': indices are split on dates where the target at t+n_forecast is in the next interval
            ex. with n_forecast = 2, freq='M':
                t       t+1   t+2
                29/01 | 30/01 31/01
                 -----------------     outer split
                30/01 | 31/01 01/02
                 -----------------     inner split
                31/01 | 01/02 02/02
        :return: tuples of (datetime, array of indices)
        """
        def round_dates(x: Tuple[int, datetime]) -> Tuple[int, datetime]:
            if boundary == 'inner':
                date_boundary = x[1] + timedelta(days=1)
            elif boundary == 'outer':
                date_boundary = x[1] + timedelta(days=self.n_forecast)
            else:
                raise ValueError(f'boundary is not a valid value. Found: {boundary}')

            if freq == 'M':
                begin_month = date_boundary.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
                rounded = x[0], begin_month
            elif freq == 'W':
                rounded = x[0], date_boundary - timedelta(days=date_boundary.weekday())
            elif freq == 'D':
                rounded = x
            else:
                raise ValueError(f'freq is not a valid value. Found: {freq}')
            return rounded

        def aggregate_dates(x, y) -> List[List[Union[List[int], datetime]]]:
            if isinstance(x, tuple):
                x = [[[x[0]], x[1]]]
            if x[-1][1] == y[1]:
                x[-1][0].append(y[0])
            else:
                x.append([[y[0]], y[1]])
            return x

        def to_np_array(x):
            for i in range(len(x)):
                x[i][0] = np.array(x[i][0])
                if format_date:
                    x[i][1] = datetime_to_str(x[i][1], freq)
            return x

        return to_np_array(reduce(aggregate_dates, map(round_dates, [(i, j) for i, j in enumerate(self.date_range)])))

    def walk_iterator(self, nb_test, periods_train=0, periods_eval=1, periods_test=1, freq='M', boundary='inner'):
        """
        iterate over indexes, giving a split for training, evaluation and test set
        :param nb_test: number of test periods included at each iteration. Default = 1 period per test set
        :param periods_train: number of periods to use in training set. 0 = use all periods at each iteration
        :param periods_eval: number of periods to use in evaluation set
        :param periods_test: total number of periods that must be evaluated in the test set
        :param freq: frequency of the split
        :param boundary: tell how to proceed for boundary: dates with values overlapping on multiple interval. cf time_idx
            for details
        :return:
        """
        time_idx = self.time_idx(freq, format_date=True, boundary=boundary)
        nb_periods = len(time_idx)
        idx_test = max(nb_periods - (periods_test * nb_test), periods_eval + periods_train)
        while idx_test < nb_periods:
            test_set = time_idx[idx_test:idx_test+periods_test]
            valid_set = time_idx[idx_test - periods_eval:idx_test]
            if periods_train == 0:
                training_set = time_idx[:idx_test - periods_eval]
            else:
                training_set = time_idx[idx_test - periods_eval - periods_train:idx_test - periods_eval]

            sets = [[training_set, periods_train], [valid_set, periods_eval], [test_set, periods_test]]
            for i in range(3):
                if sets[i][0]:
                    set_array = np.concatenate([sets[i][0][j][0] for j in range(len(sets[i][0]))])
                    if sets[i][1] > 1 or (i == 0 and sets[i][1] == 0):
                        sets[i] = set_array, f'{sets[i][0][0][1]} - {sets[i][0][-1][1]}'
                    else:
                        sets[i] = set_array, sets[i][0][-1][1]
                else:
                    sets[i] = [np.array([]), '']

            yield sets[0], sets[1], sets[2]
            idx_test += periods_test


class TestDataGenerator(unittest.TestCase):

    def setUp(self) -> None:  # called before each test example
        pass

    @classmethod
    def setUpClass(cls) -> None:  # called once before running the tests
        date_begin = "2020-02-01"
        url_world = "../data/hospi/world.csv"
        url_pop = "../data/population.txt"
        population = get_world_population(url_pop)
        renaming = {v: k for k, v in european_geocodes.items()}
        geocodes = {k: v for k, v in european_geocodes.items() if population[k] > 1_000_000}
        df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
        cls.scaler_generator = MinMaxScaler
        cls.df_hospi = df_hospi

    def test_normalisation_x(self):
        """
        test if X is normalised between 0,1 when using a MinMaxScaler(0, 1)
        """
        scaler_generator = self.scaler_generator
        df_hospi = self.df_hospi
        dg = DataGenerator(df_hospi, 20, 10, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        idx_begin = np.arange(100)
        X = dg.get_x(idx=idx_begin, scaled=True)
        self.assertAlmostEqual(1, X.max())
        self.assertAlmostEqual(0, X.min())

        idx_middle = np.arange(50, 120)
        X = dg.get_x(idx=idx_middle, scaled=True)
        self.assertAlmostEqual(1, X.max())
        self.assertAlmostEqual(0, X.min())

    def test_no_target(self):
        """
        test if a data generator can be created without target
        """
        scaler_generator = self.scaler_generator
        df_hospi = self.df_hospi
        n_forecast = 10
        dg_no_target = DataGenerator(df_hospi, 20, 0, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_no_target = dg_no_target.batch_size
        dg_target = DataGenerator(df_hospi, 20, n_forecast, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_target = dg_target.batch_size
        self.assertGreater(batch_size_no_target, batch_size_target)  # less values have been removed on a dg without target
        self.assertEqual(batch_size_no_target, batch_size_target + n_forecast)
        self.assertTrue(dg_no_target.no_target)
        self.assertFalse(dg_target.no_target)
        dg_no_target = DataGenerator(df_hospi, 20, n_forecast, '', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_no_target = dg_no_target.batch_size
        self.assertGreater(batch_size_no_target, batch_size_target)  # less values have been removed on a dg without target
        self.assertEqual(batch_size_no_target, batch_size_target + n_forecast)
        self.assertTrue(dg_no_target.no_target)

    def test_padding(self):
        """
        test if the indexes of padding are correct
        """
        msg_zero = "padding should not contain negative values"
        msg_augmented = "padding of augmented regions should be union of regions composing it"
        msg_unaugmented = "padding of unaugmented regions incorrect"
        begin_a = datetime.strptime('2020-02-01', '%Y-%m-%d').date()
        end_a = datetime.strptime('2020-06-01', '%Y-%m-%d').date()
        begin_b = datetime.strptime('2020-01-01', '%Y-%m-%d').date()
        end_b = datetime.strptime('2020-06-01', '%Y-%m-%d').date()
        begin_c = datetime.strptime('2020-02-01', '%Y-%m-%d').date()
        end_c = datetime.strptime('2020-07-01', '%Y-%m-%d').date()
        begin_d = datetime.strptime('2020-01-01', '%Y-%m-%d').date()  # longest date range
        end_d = datetime.strptime('2020-07-01', '%Y-%m-%d').date()
        date_range_a = pd.date_range(begin_a, end_a)
        date_range_b = pd.date_range(begin_b, end_b)
        date_range_c = pd.date_range(begin_c, end_c)
        date_range_d = pd.date_range(begin_d, end_d)  # longest date range
        df_dict = {
            'A': pd.DataFrame(data={'DATE': date_range_a, 'LOC': ['A' for _ in range(len(date_range_a))],
                                       'val': np.arange(len(date_range_a))}).set_index(["LOC", "DATE"]),
            'B': pd.DataFrame(data={'DATE': date_range_b, 'LOC': ['B' for _ in range(len(date_range_b))],
                                       'val': np.arange(len(date_range_b))}).set_index(["LOC", "DATE"]),
            'C': pd.DataFrame(data={'DATE': date_range_c, 'LOC': ['C' for _ in range(len(date_range_c))],
                                       'val': np.arange(len(date_range_c))}).set_index(["LOC", "DATE"]),
            'D': pd.DataFrame(data={'DATE': date_range_d, 'LOC': ['C' for _ in range(len(date_range_d))],
                                       'val': np.arange(len(date_range_d))}).set_index(["LOC", "DATE"]),
        }
        n_forecast = 10
        n_samples = 20
        days_removed = n_samples + n_forecast - 1
        dg = DataGenerator(df_dict, n_samples, n_forecast, 'val', MinMaxScaler, augment_merge=2)
        padding = dg.padded_idx_init
        for k in padding:
            self.assertCountEqual(padding[k], list(filter(lambda x: x >= 0, padding[k])), msg_zero)
        expected_padding_a = np.array([i for i, j in enumerate(date_range_d) if j > end_a])
        expected_padding_b = np.array([i for i, j in enumerate(date_range_d) if j > end_a])
        expected_padding_c = np.array([])
        expected_padding_d = np.array([])
        np.testing.assert_array_equal(padding['A'], expected_padding_a, msg_unaugmented)
        np.testing.assert_array_equal(padding['B'], expected_padding_b, msg_unaugmented)
        np.testing.assert_array_equal(padding['C'], expected_padding_c, msg_unaugmented)
        np.testing.assert_array_equal(padding['D'], expected_padding_d, msg_unaugmented)
        padding = dg.padded_idx
        for k in padding:
            self.assertCountEqual(padding[k], list(filter(lambda x: x >= 0, padding[k])), msg_zero)
        np.testing.assert_array_equal(padding['A'], expected_padding_a - days_removed, msg_unaugmented)
        np.testing.assert_array_equal(padding['B'], expected_padding_b - days_removed, msg_unaugmented)
        np.testing.assert_array_equal(padding['C'], expected_padding_c, msg_unaugmented)
        np.testing.assert_array_equal(padding['D'], expected_padding_d, msg_unaugmented)
        np.testing.assert_array_equal(padding['A-B'], expected_padding_a - days_removed, msg_augmented)
        np.testing.assert_array_equal(padding['A-C'], expected_padding_a - days_removed, msg_augmented)
        np.testing.assert_array_equal(padding['A-D'], expected_padding_a - days_removed, msg_augmented)
        np.testing.assert_array_equal(padding['B-C'], expected_padding_b - days_removed, msg_augmented)
        np.testing.assert_array_equal(padding['B-D'], expected_padding_b - days_removed, msg_augmented)
        np.testing.assert_array_equal(padding['C-D'], expected_padding_c, msg_augmented)


if __name__ == "__main__":
    unittest.main()

