import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import arrow
import datetime
import os
import tabulate
import sys
import time
import concurrent.futures

#grab stock data
SYMBOL_NAME_FILE = "Stock2.txt"
RESULT_FILE = "Result.csv"
GlobalTimeFrame=0


DataTime = [
            ["5y", "1d"],
            ["1y", "1d"],
            ["1y", "1h"],
            ["5d", "1d"]
            ]

class Stock:

    def __init__(self, symbol='SPY', data_range='5d', data_interval='1m'):
        self.symbol = symbol
        self.data_range = data_range
        self.data_interval = data_interval
    
    def get_data(self):
	    #"validRanges":["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]},
	    #Only 7 days worth of 1m granularity data are allowed to be fetched per request.
	    #interval=4h is not supported. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]"}}}
		
	    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval={}'.format(self.symbol,self.data_range,self.data_interval))
	    data = res.json()
	    body = data['chart']['result'][0]    
	    dt = datetime.datetime
	    dt = pd.Series(map(lambda x: arrow.get(x).datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
	    self.dataframe = pd.DataFrame(body['indicators']['quote'][0], index=dt)
	    dg = pd.DataFrame(body['timestamp'])    
	    self.dataframe = self.dataframe.loc[:, ('open', 'high', 'low', 'close', 'volume')]
	    self.dataframe.dropna(inplace=True)     #removing NaN rows
	    self.dataframe.columns = ['OPEN', 'HIGH','LOW','CLOSE','VOLUME']    #Renaming columns in pandas

	    return self.dataframe

	#sma function
    def SMA(self, period=30):
	    return self.dataframe['CLOSE'].rolling(window=period).mean()

    def EMA(self, period=30):
	    return self.dataframe['CLOSE'].ewm(span=period).mean()
	
    def RSI(self, n=14):
        chg = self.dataframe['CLOSE'].diff(1)
        gain = chg.mask(chg<0,0)
        loss = chg.mask(chg>0,0)

        avg_gain = gain.ewm(com = n-1, min_periods=n).mean()
        avg_loss = loss.ewm(com = n-1, min_periods=n).mean()
        rs = abs(avg_gain/avg_loss)
        rsi = 100 - (100/(1+rs))     
        return rsi

    def MACD(self,symbol):
        """
        compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
        return value is emaslow, emafast, macd which are len(x) arrays
        """
        emaslow = symbol.EMA(26)
        emafast = symbol.EMA(12)
        return emafast - emaslow, symbol.EMA(9)

def Diff(list1, list2): 
    zip_object = zip(list1, list2)
    difference = []
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)
    return difference


def LongStats(symbol, data):
    summary = {'buy':[], 'sell':[], 'profit':[]}
    flag = -1

##    data1 = symbol.SMA(15)
##    data2 = symbol.SMA(50)
    
##    LTT = symbol.EMA(200)
##    data2 = symbol.EMA(sma2)
    
    data3 = symbol.RSI()
    
    for i in range(len(data)):

        #SMA & EMA
##        BuySignal = True if data1[i] > data2[i] else False
##        SellSignal = True if data1[i] < data2[i] else False

##        BuySignal = True if data['CLOSE'][i] > data2[i] else False
##        SellSignal = True if data['CLOSE'][i] < data2[i] else False
        
##        BuySignal = True if MACD1[i] > MACD2[i] else False
##        SellSignal = True if MACD1[i] < MACD2[i] else False
        
        #RSI
        BuySignal = data3[i] < 20
        SellSignal = data3[i] > 80
##        
        if BuySignal:
            if flag !=1:
                summary['buy'].append(data['CLOSE'][i])
                flag =1
        elif SellSignal:
            if flag == -1:
                continue
            if flag !=0:
                summary['sell'].append(data['CLOSE'][i])
                flag = 0
                

    if flag !=0:
        summary['buy'] = summary['buy'][:-1]
        
    summary['profit']= out = np.divide(Diff(summary['sell'], summary['buy']), summary['buy'])*100
    


    pos_count = 0
    neg_count = 0
    TGain = 0
    TLoss = 0
    
    for num in summary['profit']: 
        # checking condition 
        if num > 0: 
            pos_count += 1
            TGain += num
        else:
            neg_count += 1
            TLoss += num

    RGain = "0.00"
    RLoss = "0.00"
    
    if pos_count != 0:
        RGain = "{0:.4f}".format(TGain/pos_count)
        
    if neg_count != 0:    
        RLoss = "{0:.4f}".format(TLoss/neg_count)
    
    if len(summary['profit']) != 0:
        return "{0:.2f}".format(pos_count/len(summary['profit']) * 100),RGain , RLoss, "{0:.4f}".format(len(summary['profit']))

    return "{0:.2f}".format(0), "{0:.4f}".format(0), "{0:.4f}".format(0), "{0:.4f}".format(0)


def GetSymbolFromCSV(myFilePath):
    content_array = []
    with open(myFilePath, 'r') as file:
        for line in file:
            content_array.append(line.strip('\n\r,'))
    return content_array

def WriteAnalyzeResult(results_array):
    file = open(RESULT_FILE, 'w+')
    ouput = pd.DataFrame(results_array)
    ouput.to_csv(RESULT_FILE, index=False)


def StockTimeframes(StocksName):
    switcher={
            0: Stock(StocksName, '5y', '1d'),
            1: Stock(StocksName,'1y', '1h'),
            2: Stock(StocksName,'5d', '1m'),  
        }
    return switcher.get(GlobalTimeFrame,"Invalid timeframe")
    
def DataAnalysis(StocksName):
    
    AnalyzeStocksData = {'Stock':StocksName, 'Gain':'', 'Loss':'', "Percent":'', "Entry":''}
 
    TargetSymbol = StockTimeframes(StocksName)
    
    StockPriceData = TargetSymbol.get_data()
    percentage,gain, loss, entry = LongStats(TargetSymbol, StockPriceData)
    
    AnalyzeStocksData['Percent'] = percentage
    AnalyzeStocksData['Gain']= gain
    AnalyzeStocksData['Loss']= loss
    AnalyzeStocksData['Entry']= entry
    
    return AnalyzeStocksData

def DataSummary(results_array):

    TotalGain = 0
    GainCount = 0
    
    TotalLoss = 0
    LossCount = 0
    
    TotalEntry = 0
    EntryCount = 0
    
    TotalWinPercent = 0
    length = 0
    
    for i in range(len(results_array)):

        if results_array[i]['Gain'] != "0.00":
            GainCount += 1
       
        if results_array[i]['Loss'] != "0.00":
            LossCount += 1

        if results_array[i]['Entry'] != "0.00":
            EntryCount += 1
            
        TotalGain += float(results_array[i]['Gain'])
        TotalLoss += float(results_array[i]['Loss'])
        TotalWinPercent += float(results_array[i]['Percent'])
        TotalEntry += float(results_array[i]['Entry'])

    FinalGain = 0
    FinalLoss = 0
    FinalPercent = 0
    
    if GainCount != 0:
        FinalGain = TotalGain/GainCount
        
    if LossCount != 0:    
        FinalLoss = TotalLoss/LossCount   

    if EntryCount != 0:
        FinalPercent = TotalWinPercent/EntryCount
        
    return "{0:.4f}".format(FinalGain), "{0:.4f}".format(FinalLoss), "{0:.2f}".format(FinalPercent),"{}".format(TotalEntry)

##    print("average gain:{0:.2f}".format(FinalGain))
##    print("average loss:{0:.2f}".format(FinalLoss))
##    print("average win %:{0:.2f}".format(FinalPercent))
##    print("total entry:{}".format(TotalEntry))
##    print("Count Gain: {0:.2f}".format(GainCount))
##    print("Count Lost: {0:.2f}".format(LossCount))

def TestCode():
    TargetSymbol = Stock("AAPL","1y","1d")
    TargetSymbol.get_data()
    SMA = TargetSymbol.SMA(20)
    EMA = TargetSymbol.EMA(20)
    RSI = TargetSymbol.RSI()
    print(SMA)
    print(EMA)
    print(RSI)

def TestCode2():
    result = DataAnalysis("AAPL")
    
    
if __name__ == '__main__':

    StartTime = time.perf_counter()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)


##    Symbols = GetSymbolFromCSV(SYMBOL_NAME_FILE)
    Symbols = ["AMD"]
    
    TableResult = {"TIMEFRAME":[],"WIN PERCENT":[], "AVG GAIN":[],"AVG LOSS":[],"ENTRIES":[]}
    for i in range(0,3):
        GlobalTimeFrame = i
        Title = {
                0: "1d timeframe, 5y range",
                1: "1h timeframe, 1y range",
                2: "1m timeframe, 5d range"
            }
        TableResult["TIMEFRAME"].append(Title.get(i, "invalid timeframe"))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(DataAnalysis, Symbols)

        result_array = []
        for result in results:
            result_array.append(result)
            
        AG,AL, WP, ENTR= DataSummary(result_array)
        TableResult["AVG GAIN"].append(AG)
        TableResult["AVG LOSS"].append(AL)
        TableResult["WIN PERCENT"].append(WP)
        TableResult["ENTRIES"].append(ENTR)
        #WriteAnalyzeResult(result_array)

    print(pd.DataFrame(TableResult))
    
    FinishTime = time.perf_counter()
    print(f'Finished in {FinishTime-StartTime} seconds')

##    TestCode()

##    TestCode2()


    

