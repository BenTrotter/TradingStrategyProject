import matplotlib.pyplot as plt
import csv
from datetime import datetime
import pandas as pd
import numpy as np

data = pd.read_csv('MSFT.csv')

data['Data'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

price_date = data['Date']
price_close = data['Close']
# price_high = data['High']
# price_low = data['Low']
rsi = data['RSI 14']
#ma15 = data['SMA 10.0']
#ma30 = data['SMA 12.0']

fig, ax = plt.subplots()

plt.plot_date(price_date, rsi, 'r--', label='RSI 14')
#plt.plot_date(price_date, ma15, 'b--', label='15-day moving avg')
#plt.plot_date(price_date, ma30, 'r--', label='30-day moving avg')
plt.plot_date(price_date, price_close, 'k', label='Closing Price')
#plt.plot_date(price_date, price_low, linestyle='solid', label='Low')
plt.legend()

every_nth = 12
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

plt.gcf().autofmt_xdate()

plt.title('Microsoft Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()
