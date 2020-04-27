import itertools

import pytrends
from pytrends.request import TrendReq
import pandas as pd
import time
import datetime
from datetime import datetime, date, time
import matplotlib.pyplot as plt
import seaborn as sns

result = []
filename = 'symptoms.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for i in lines:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = []
        if len(i) > 1:
            kw_list.append(i[0:len(i)-1])
            pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-04-26', geo='BE', gprop='')
            interest_over_time_df = pytrends.interest_over_time()
            mean = interest_over_time_df.mean(axis=0)
            result.append((i, mean[0]))

result.sort(key=lambda x: x[1], reverse=True)
print(result)

pytrends = TrendReq(hl='en-US', tz=360)

kw_list = []
for item in itertools.islice(result, 5):
    kw_list.append(item[0])
print(kw_list)

pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-04-23', geo='BE', gprop='')

interest_over_time_df = pytrends.interest_over_time()
print(interest_over_time_df)


# Drawing of the interest of the keywords over time
sns.set(color_codes=True)
dx = interest_over_time_df.plot.line(figsize=(9,6), title = "Interest over time in Belgium")
dx.set_xlabel('Date')
dx.set_ylabel('Trends index')
dx.tick_params(axis='both', which='major', labelsize=13)
plt.show()