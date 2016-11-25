from flask import Flask
from flask import request, url_for, redirect, session, escape, abort
from flask import render_template
app = Flask(__name__)
app.secret_key= "ZDFGHDHDFGHJGHK#@#$R%TWDFGSDFASDA"
import argparse
import json
from sim import *
from collections import OrderedDict

@app.route('/', methods=['POST', 'GET'])
def Main():
    #filter user here
    return render_template('display.html')

@app.route('/data', methods=['POST','GET'])
def Data():
    stock=request.form.get("stock","apple")
    direction=request.form.get("direction","1")
    if int(direction)!=-1:
        direction=1
    else:
        direction=-1
    mode=request.form.get("mode","chase")
    logger.info("Mode is {}".format(mode))
    start=int(request.form.get("start_year",2014))
    end=int(request.form.get("end_year",2016))
    starti=int(request.form.get("start_index",0))
    endi=int(request.form.get("end_index",100))
    funds=float(request.form.get("param_funds",100000))
    units=float(request.form.get("param_units",4000))
    logger.info(request.form)
    if stock=="apple":
        S=ThresholdTurningPointSimulatorApple(funds=funds,u=units,start_year=start, start_index=starti, end_index=endi, end_year=end, kws=request.form)
    else:
        S=ThresholdTurningPointSimulatorTwitter(funds=funds,u=units,start_year=start, end_year=end,start_index=starti, end_index=endi,  kws=request.form)
    output={}
    data={}
    data["cols"]=[
            {"id":"","label":"Time","pattern":"","type":"datetime"},
            {"id":"","label":"Price","pattern":"","type":"number"},
            ]
    data["rows"]=[]
    for t,p in zip(S.ts, S.prices):
        d="Date({},{},{},{})".format(t.year,t.month,t.day,t.hour)
        data["rows"].append({"c":[{"v":d, "f":None},{"v":p,"f":None}]})
    output["chart"]=data

    data={}
    data["cols"]=[
            {"id":"","label":"Time","pattern":"","type":"datetime"},
            {"id":"","label":"Funds","pattern":"","type":"number"},
            {"id":"","label":"Units","pattern":"","type":"number"},
            {"id":"","label":"Gains","pattern":"","type":"number"},
            ]
    data["rows"]=[]
    for t,p1,p2,p3 in zip(S.ts, S.funds,S.units, S.gains):
        d="Date({},{},{},{})".format(t.year,t.month,t.day,t.hour)
        data["rows"].append({"c":[{"v":d, "f":None},{"v":p1,"f":None},{"v":p2,"f":None},{"v":p3,"f":None}]})
    output["chart2"]=data

    data={}
    data["cols"]=[
            {"id":"","label":"Time","pattern":"","type":"datetime"},
            {"id":"","label":"Price","pattern":"","type":"number"},
            {"id":"","label":"Amount","pattern":"","type":"number"},
            {"id":"","label":"Source","pattern":"","type":"string"},
            ]
    data["rows"]=[]
    for row in S.orders:
        t=row[0]
        d1="Date({},{},{},{})".format(t.year,t.month,t.day,t.hour)
        data["rows"].append({"c":[{"v":d1,"f":None},
            {"v":row[1],"f":None},{"v":row[2],"f":None},{"v":row[3],"f":None}]})
    output["table"]=data

    return json.dumps(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web Server for AqiClient(Simulator)')
    parser.add_argument('--port', type=int, help='Port number for deployment')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    app.run(host='0.0.0.0',port=args.port,debug=args.debug)
