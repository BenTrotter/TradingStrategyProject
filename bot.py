import numpy as np
import csv
import math
import pandas as pd
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
import schedule
import time
from preload import load
import operator

import deap
from importlib import reload
reload(deap)

from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import alpaca_trade_api as tradeapi
alpaca_endpoint = 'https://paper-api.alpaca.markets'
api = tradeapi.REST('PKAMX78YJ23D54CMUHU9','ICwqGguOIpxZgQ7KKcV8gdjGR37RJvrVkxv9gaiV',alpaca_endpoint)

# Global
file = "MSFT.csv"
ticker = "MSFT"
global position
position = False
global email
email = True
global trade
trade = False
global password
password = "Bot123Win" # hide if going live
global numShares
numShares = 4

class pd_float(object):
    pass

class pd_bool(object):
    pass
class position_bool(object):
    pass

class pd_int(object):
    pass

class ema_int(object):
    pass

class ma_int(object):
    pass

class rsi_int(object):
    pass
class rsi_bounds(object):
    pass
class rsi_ans(object):
    pass
class rsi_lt(object):
    pass
class rsi_gt(object):
    pass

class macd_sig(object):
    pass
class macd_f(object):
    pass
class macd_s(object):
    pass
class macd_ans(object):
    pass
class macd_bounds(object):
    pass

class so_int(object):
    pass
class so_ans(object):
    pass
class so_bounds(object):
    pass

def ma(date,window):

    panda = pd.read_csv(file) # Maybe set MSFT.csv variable to a global variable
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):

    panda = pd.read_csv(file)
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):

    panda = pd.read_csv(file)
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):

    panda = pd.read_csv(file)
    macd = panda["MACD "+str(s)+str(f)]
    sig = panda["MACD Signal "+str(sig)+str(s)+str(f)]
    dates = panda["Date"]
    dict1 = dict(zip(dates,macd))
    dict2 = dict(zip(dates,sig))
    macd1 = dict1[date]
    sig1 = dict2[date]
    ans = macd1 - sig1
    return ans

def so(date,window):

    panda = pd.read_csv(file)
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)] # Maybe make this a variable that is explored by the algorithm
    dates = panda["Date"]
    dict1 = dict(zip(dates,SO))
    dict2 = dict(zip(dates,sig))
    so = dict1[date]
    sig1 = dict2[date]
    ans = so - sig1 # Check this is the right way round. It may not matter.
    return ans

def if_then_else(input, output1, output2):
    return output1 if input else output2

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", [str,position_bool], pd_bool)
# boolean operators
pset.addPrimitive(if_then_else, [position_bool, rsi_gt, rsi_lt], pd_bool)
pset.addPrimitive(operator.and_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.or_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.not_, [pd_bool], pd_bool)
pset.addPrimitive(operator.lt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.gt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.lt, [rsi_ans,rsi_bounds], rsi_lt)
pset.addPrimitive(operator.gt, [rsi_ans, rsi_bounds], rsi_gt)
pset.addPrimitive(operator.lt, [macd_bounds, macd_ans], pd_bool) # for above or below zero, maybe make macd_bounds and so_bounds the same thing
pset.addPrimitive(operator.gt, [macd_bounds, macd_ans], pd_bool)
pset.addPrimitive(operator.lt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(operator.gt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(ma, [str, ma_int], pd_float)
pset.addPrimitive(ema, [str, ema_int], pd_float)
pset.addPrimitive(rsi, [str, rsi_int], rsi_ans)
pset.addPrimitive(macd, [str, macd_s, macd_f, macd_sig], macd_ans)
pset.addPrimitive(so, [str, so_int], so_ans)

def job():

    global position

    today = datetime.today()
    nextDay = today + timedelta(days=1)
    date = today.strftime("%Y-%m-%d")
    nextDate = nextDay.strftime("%Y-%m-%d")

    load(ticker,"2021-01-01", nextDate, '1d')

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    
    try:
        i = operator.or_(operator.lt(ema(date, 12), ema(date, 26)), operator.or_(operator.gt(0, so(date, 7)), operator.gt(ema(date, 200), ma(date, 10))))
        rule = toolbox.compile(expr=i)

        action = rule(date,position)

        if action and position == False:
            buy = True
            sell = False
        elif not action and position == False:
            sell = False
            buy = False
        elif action and position == True:
            sell = False
            buy = False
        elif not action and position == True:
            sell = True
            buy = False

        if position:
            msg1 = "You currently own "+str(numShares)+" shares of Microsoft."
        else:
            msg1 = "You currently do not own any shares of Microsoft."

        if buy:
            print("Buy")
            position = True
            if email:
                msg = "buy"

            if trade:
                api.submit_order(symbol='MSFT',
                qty=numShares,
                side='buy',
                time_in_force='gtc',
                type='market'
                )

        elif sell:
            print("Sell")
            position = False
            if email:
                msg = "sell"

            if trade:
                api.submit_order(symbol='MSFT',
                qty=numShares,
                side='sell',
                time_in_force='gtc',
                type='market'
                )

        else:
            print("Hold")
            if email:
                msg = "hold"

        if email:
            port = 465  # For SSL
            smtp_server = "smtp.gmail.com"
            sender_email = "pythonstrategyupdate@gmail.com"  # Enter your address
            receiver_email = "benjamin.trotter@yahoo.co.uk"  # Enter receiver address

            message = MIMEMultipart("alternative")
            message["Subject"] = "Daily Strategy Update - "+ticker
            message["From"] = sender_email
            message["To"] = receiver_email

            html = """\
            <html>
            <body>
                <p>Good Evening Benjamin,<br><br>
                I hope you've had a lovely day<br><br>
                """+msg1+"""
                <br><br>
                We have run the checks on your strategy since todays close
                and have decided to """+msg+""" the stock. Alpaca should have
                automatically executed this action for you.
                </p>
                <p>All the best,<br>Trading Bot</p>
            </body>
            </html>
            """

            part1 = MIMEText(html, "html")

            # Add HTML/plain-text parts to MIMEMultipart message
            # The email client will try to render the last part first
            message.attach(part1)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message.as_string())

    except KeyError:
        print("No trading today")

    return

schedule.every().day.at("13:06").do(job)

while True:
    schedule.run_pending()
    time.sleep(30) # wait one hour




