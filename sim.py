#Simulation routines for testing the algorithms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import time
import pandas as pd
from strategy import *

def ThresholdTurningPointSimulatorApple(funds=100000,u=400, start_year=2014, end_year=2016, start_index=0, end_index=100, kws={}):

    apple_data=pd.read_csv("./apple.csv")
    apple_data.sort("date",inplace=True)

    apple_inst = Instrument(1,"Apple","APPL", "stock")
    cnt=0
    for t, start, stop in zip(apple_data["date"],apple_data["open"],apple_data["close"]):
        if "/" not in t:
            continue
        year=int(t.split("/")[0])
        if (year<start_year) or (year>end_year):
            continue
        if cnt==0:
            t2=t+" 9:30"
            apple_inst.update(t2,float(start))
            S=PrototypeStrategyI(apple_inst, funds, u, 200,**kws)
            cnt+=1
            continue
        cnt+=1
        if cnt <start_index or cnt > end_index:
            continue
        t2=t+" 9:30"
        apple_inst.update(t2,float(start))
        S.update()
        t2=t+" 16:00"
        apple_inst.update(t2,float(stop))
        S.update()
    logger.info("Current fund is {}".format(str(S.fund)))
    logger.info("Current unit is {}".format(S.unit))
    return S

def ThresholdTurningPointSimulatorTwitter(funds=100000,u=400, start_year=2014, end_year=2016,start_index=0, end_index=100, kws={}):

    twitter_data=pd.read_csv("./twitter.csv")
    twitter_data.sort("Date",inplace=True)

    twitter_inst = Instrument(2, "Twitter","TWIT", "stock")
    cnt=0
    for t, start, stop in zip(twitter_data["Date"],twitter_data["Open"],twitter_data["Close"]):
        if "-" not in t:
            continue
        year=int(t.split("-")[0])
        if year < start_year or year>end_year:
            continue
        if cnt==0:
            t2=t+" 9:30"
            twitter_inst.update(t2,float(start))
            S=PrototypeStrategyI(twitter_inst, funds, u, 200, **kws)
            cnt+=1
            continue
        cnt+=1
        if cnt <start_index or cnt > end_index:
            continue
        t2=t+" 9:30"
        twitter_inst.update(t2,float(start))
        S.update()
        t2=t+" 16:00"
        twitter_inst.update(t2,float(stop))
        S.update()
    logger.info("Current fund is {}".format(str(S.fund)))
    logger.info("Current unit is {}".format(S.unit))
    # logger.info("Overall profit (if clearing now) is {}".format(str(S.funds+S.units*S.inst.price-100000)))
    return S


if __name__=="__main__":
    ThresholdTurningPointSimulatorApple()
