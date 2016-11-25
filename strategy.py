import datetime
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Strategy Simulator")
logger.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class Instrument(object):
    """
    Instrument is a stock or future, it associates with itself
    data such as prices and update time
    """
    def __init__(self, id, name, sym, itype="stock"):
        if itype not in ["future","stock"]:
            raise ValueError("Type of Instrument is not supported : {}".format(itype))
        if type(id) != int:
            raise ValueError ("Your instrument id is invalid {}".format(id))
        self.type = itype
        self.id = id
        self.current_raw = {}
        self.data={}
        self.symbol=sym
        self.name=name
        self.ts=None
        self.prices = pd.Series()
        self.holdings=0
        return

    def update(self,ts, price):
        self.data["price"]=price
        self.ts=pd.Timestamp(ts)
        self.prices = self.prices.append(pd.Series(self.price, [self.ts]))
        return

    @property
    def price(self):
        return self.data.get("price",float('nan'))

    def __unicode__(self):
        return u"[I][{type},{id}][{sym}] {name}: {price} @ {time}, {low}~{high}".format(type=self.type,
                id=self.id, sym=self.symbol, price=self.price, name=self.name,
                time=self.ts, low=self.data.get("lowPrice",float('nan')),high=self.data.get("highPrice",float('nan')))

    def __str__(self):
        return self.__unicode__().encode("utf-8")


class Subroutine(object):
    """
    Subroutines are modular algorithms that have
    internal states and iteract with the main alglrithm
    through input of prices/time and output of decisions
    init -> update (price/time) * N ->  output (terminate, buy/sell)
    importantly, a subroutine traces one instrument
    """
    def __init__(self, inst, unit,**kwargs):
        self.inst=inst
        self.unit=unit
        self.init(**kwargs)

    def init(self,**kwargs):
        for k, v in kwargs.items():
            setattr(self,k,v)
        return

    def update(self, instUpdate=False):
        pass

    def output(self):
        """
        none -> terminated
        positive amount -> buy
        negative amount -> sell
        """
        return None

class ThresholdControl(Subroutine):
    """
    This subroutine aims to reduce risk by moving towards neutral units holding
    influenced by the gain at present
    """
    def __init__(self, inst, fund, unit, winningPer=0.4, losingPer=0.2,
            sellingPerWin=0.5, sellingPerLose=1, **kwargs):
        self.inst=inst
        self.prices=[self.inst.price]
        self.funds=[fund]
        self.units=[unit]
        self.total0=fund+unit*self.inst.price
        self.times=[self.inst.ts]
        self.winningPer=float(winningPer)
        if self.winningPer>1:
            raise ValueError("Winning Per incorrect")
        self.losingPer=float(losingPer)
        if self.losingPer>1:
            raise ValueError("Losing Per incorrect")
        self.sellingPerWin=float(sellingPerWin)
        self.sellingPerLose=float(sellingPerLose)
        if self.sellingPerWin>1 or self.sellingPerLose>1:
            raise ValueError("Selling Per incorrect")

    def update(self, fundIn, deltaUnit=0):
        """
        When updates returns a non-zero number
        the program should also be terminated or restarted as seen fit
        """
        self.prices.append(self.inst.price)
        self.times.append(self.inst.ts)
        fund=fundIn
        unit=self.units[-1]+deltaUnit
        if unit==0:
            return 0
        total=fund+unit*self.prices[-1]
        gain=total-self.total0
        # logger.info("Current gain {}, total {} fund {} unit {}".format(gain,total, fund, unit))
        if gain>=0 and gain/self.total0>=self.winningPer:
            # sellingUnit = abs(gain)*self.sellingPerWin/self.inst.price
            sellingUnit = abs(unit)*self.sellingPerWin
            if sellingUnit>=abs(unit):
                sellingUnit=abs(unit)
            if unit >0:
                sellingUnit=-sellingUnit
            logger.info("Winning Control: gain={}({}), selling {} units".format(gain, gain/self.total0, sellingUnit ))
            self.units.append(unit+sellingUnit)
            self.funds.append(fund+sellingUnit*self.inst.price)
            return sellingUnit
        elif gain<=0 and abs(gain/self.total0)>=self.losingPer:
            # sellingUnit = abs(gain)*self.sellingPerLose/self.inst.price
            sellingUnit = abs(unit)*self.sellingPerLose
            if sellingUnit>=abs(unit):
                sellingUnit=abs(unit)
            if unit>0:
                sellingUnit=-sellingUnit
            logger.info("Losing Control: gain={}({}), selling {} units".format(gain, gain/self.total0, sellingUnit))
            self.units.append(unit+sellingUnit)
            self.funds.append(fund+sellingUnit*self.inst.price)
            return sellingUnit
        else:
            self.funds.append(fund)
            self.units.append(unit)
        return 0

class Chasing(Subroutine):
    """
    This strategy aims to chase the larger scale trend of the instrument
    It should be run with threshold control to manage risks
    There are two modes, one is to increase holding in long term growing market
    The other is to sell in a growing market at turning points
    It requires an initial unit investment to start working, that investment is not evaluated
    Currently it does not consider shorting
    """
    def __init__(self, inst, unit, trend = 1, mode_chase="chase", gap=5, upper_limit=1,lower_limit=0.5,
           init=50, inc=10, safetyamount=20, **kwargs):
        if mode_chase not in ["chase","safety"]:
            # it has two modes: chasing (increasing holding) or selling (increasing fund)
            raise ValueError("Unsupported mode of operation {}".format(mode_chase))
        logger.info("init is {}".format(init))
        self.mode=mode_chase
        if trend!=-1:
            trend=1
        self.trend=trend # trend indicates whether or not the market is expected to go up (1) or down
        self.inst=inst
        self.prices=[self.inst.price]
        self.times=[self.inst.ts]
        # self.unit=unit
        # self.units=[unit]
        self.total=self.prices[-1]*unit
        self.direction=0
        self.high=-10000
        self.low=10000
        self.init=init
        self.inc=inc
        self.safetyamount=safetyamount
        self.stack=0
        self.buysell=0
        self.upper_limit=upper_limit
        self.lower_limit=lower_limit
        self.gap=gap
        self.nextStepUp=self.inst.price+gap
        self.nextStepDown=self.inst.price-gap
    def update(self):
        price=self.inst.price
        time=self.inst.ts
        price0=self.prices[-1]
        self.prices.append(price)
        self.times.append(time)
        if price > price0:
            direction=1
        elif price < price0:
            direction=-1
        else:
            return 0
        if self.direction==0:
            self.direction=direction
            return 0
        if self.direction>0 and direction<0:
            self.high=price0
        elif self.direction<0 and direction>0:
            self.low=price0
        self.direction=direction
        amount = 0
        if self.mode=="chase":
            if self.trend==1:
                if price >= self.nextStepUp:
                    self.buysell=1
                if price >= self.nextStepUp + self.upper_limit:
                    if self.buysell==1:
                        amount = self.init - self.stack*self.inc
                        if amount <= 0:
                            amount=0
                        else:
                            self.stack+=1
                        self.nextStepUp=price+self.gap
                        self.buysell=0
                elif price <= self.nextStepUp - self.lower_limit:
                    self.buysell=0
            else:
                if price <= self.nextStepDown:
                    self.buysell=-1
                if price <= self.nextStepDown - self.lower_limit:
                    if self.buysell==-1:
                        amount = self.init - self.stack*self.inc
                        amount = -amount
                        if amount >= 0:
                            amount=0
                        else:
                            self.stack+=1
                        self.nextStepDown=price-self.gap
                        self.buysell=0
                elif price >= self.nextStepDown + self.upper_limit:
                    self.buysell=0
        else:
            if self.trend==1:
                if self.buysell==0:
                    if price >= self.nextStepUp:
                        self.buysell=-1
                        self.nextStepUp = price+self.gap
                        self.nextStepDown = price-self.gap
                    elif price <= self.nextStepDown:
                        self.buysell=-1
                        self.nextStepDown= price -self.gap
                elif self.buysell==-1 and direction==-1 and self.high - price >= self.lower_limit:
                    amount = -self.safetyamount
                    self.buysell=0
            else:
                if self.buysell==0:
                    if price <= self.nextStepDown:
                        self.buysell=1
                        self.nextStepUp = price+self.gap
                        self.nextStepDown = price-self.gap
                    elif price >= self.nextStepUp:
                        self.buysell=1
                        self.nextStepUp= price + self.gap
                elif self.buysell==1 and direction==1 and  price-self.low >= self.lower_limit:
                    amount = self.safetyamount
                    self.buysell=0
                pass
        return amount




class TurningPoint(Subroutine):
    """
    This strategy aims to micromanage small scale fluctuations
    the goal can be to increase fund, increase or decrease holdings,
    or to increase hand size
    """

    def __init__(self, inst, mode_turning="increase", buysell=-1,n=10,
            n_delta=1, h=1.0, **kwargs):
        self.inst=inst
        self.buysell=buysell # this variable controls the next move 1 is buy -1 is sell
        self.pt=self.inst.price
        self.prices=[self.pt]
        self.times=[self.inst.ts]
        self.highs=[]
        self.n_delta=n_delta
        self.h=h
        self.lows=[]
        self.gain=0
        self.gains=[]
        if mode_turning not in ["increase", "decrease", "size"]:
            raise ValueError("Mode error {}".format(mode_turning))
        self.mode=mode_turning
        self.selling=n
        self.buying=n
        self.cnt=0
        self.direction=0
        self.high=-10000
        self.low=10000
        return

    def update(self):
        price0=self.prices[-1]
        price=self.inst.price
        self.prices.append(price)
        self.times.append(self.inst.ts)
        if price > price0:
            direction=1
        elif price < price0:
            direction=-1
        else:
            return 0
        if self.direction==0:
            # this covers the case for the first update
            self.direction=direction
            return 0
        if self.direction>0 and direction<0:
            self.high=price0
            self.highs.append(self.high)
        elif self.direction<0 and direction>0:
            self.low=price0
            self.lows.append(self.low)
        self.direction=direction
        amount =0
        if self.buysell==1 and direction==1 and price-self.low >= self.h:
            amount=self.buying
            self.gain-=amount*price
            self.buysell=-1
        elif self.buysell==-1 and direction==-1 and self.high - price >= self.h:
            amount=-self.selling
            self.gain-=amount*price
            self.buysell=1
        self.gains.append(self.gain)
        if amount!=0:
            # if self.gain<0:
                # return amount
            self.cnt+=1
            n_delta=self.n_delta
            if n_delta > self.gain/price:
                n_delta=int(self.gain/price)
            logger.info("Gain is {}, delta is {}, {}".format(self.gain,n_delta,self.gain/price))
            if n_delta<0:
                n_delta=0
            if self.mode=="increase":
                self.buying+=n_delta
            elif self.mode=="decrease":
                self.selling+=n_delta
            else:
                self.buying+=n_delta
                self.selling+=n_delta
        return amount



class PrototypeStrategyI(object):
    """
    This is a prototype strategy that centers around 3 subroutines:
    1, threshold management
    2, (low resolution) decremental chasing
    3, (high resolution) turning point strategy
    """
    def __init__(self, inst, fund, unit, unitInit,mode="chase", **kwargs):
        if type (mode)==list:
            mode=mode[0]
        if mode not in ["chase","turning"]:
            raise ValueError("Mode not supported {}".format(mode))
        self.mode=mode
        self.inst=inst
        self.fund=fund
        self.unit=unit
        self.total0=fund+unit*self.inst.price
        self.unitInit=unitInit
        self.ts=[]
        self.funds=[]
        self.units=[]
        self.gains=[]
        self.orders=[]
        self.prices=[]
        self.kwargs={}
        for k,v in kwargs.items():
            try:
                self.kwargs[k]=float(v[0])
            except:
                try:
                    self.kwargs[k]=str(v[0])
                except:
                    logger.info("Failed to convert {},{}".format( k,v))
        logger.info(self.kwargs)
        self.ThresholdControlRoutine=ThresholdControl(inst, fund, unit, **self.kwargs)
        self.ChasingRoutine=Chasing(inst, unitInit, **self.kwargs)
        self.TurningPointRoutine=TurningPoint(inst, **self.kwargs)
    def transact(self,n,src="other"):
        """
        negative n=> selling
        """
        if self.fund - self.inst.price*n <= 0:
            logger.warn("Running out of funds when trying to transact {}".format(n))
            newn=self.fund/self.inst.price
            self.unit+=newn
            self.fund=0
        else:
            self.fund-=self.inst.price*n
            self.unit+=n
        self.orders.append((self.inst.ts, self.inst.price, n, src))
    def update(self):
        # logger.info("Current: fund {}, unit {}, price {}".format(self.fund, self.unit, self.inst.price))
        self.ts.append(self.inst.ts)
        self.prices.append(self.inst.price)
        self.funds.append(self.fund)
        self.units.append(self.unit)
        self.gains.append(self.fund+self.unit*self.inst.price-self.total0)
        if self.mode=="chase":
            n= self.ChasingRoutine.update()
        else:
            n=self.TurningPointRoutine.update()
        n2=self.ThresholdControlRoutine.update(self.fund, n)
        if n2==0:
            if n!=0:
                self.transact(n, src=self.mode)
        else:
            logger.info("n2 is {}".format(n2))
            self.transact(n2, src="thresholdcontrol")
            self.ThresholdControlRoutine=ThresholdControl(self.inst, self.fund, self.unit, **self.kwargs)
            #restart the control here
        return
