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

#grab stock data
def get_quote_data(symbol='AAPL', data_range='1y', data_interval='1h'):
    #"validRanges":["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]},
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    body = data['chart']['result'][0]    
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])    
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    df.dropna(inplace=True)     #removing NaN rows
    df.columns = ['OPEN', 'HIGH','LOW','CLOSE','VOLUME']    #Renaming columns in pandas

    return df

#sma function
def SMA(smadf, period=30):
    return smadf['CLOSE'].rolling(window=period).mean()


#Signal Function to buy or sell
def buy_sell(data, sma1, sma2):
    sigPriceBuy=[]
    sigPriceSell=[]
    flag = -1

    for i in range(len(data)):
        smadata1 = SMA(df,sma1)[i]
        smadata2 = SMA(df,sma2)[i]
        
        if smadata1 > smadata2:
            if flag !=1:
                sigPriceBuy.append(data['CLOSE'][i])
                sigPriceSell.append(np.nan)
                flag =1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
                flag = 1
        elif smadata1 < smadata2:
            if flag !=0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data['CLOSE'][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)

def Diff(list1, list2): 
    zip_object = zip(list1, list2)
    difference = []
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)
    return difference


def buy_sell_win_long(data, sma1, sma2):
    summary = {'buy':[], 'sell':[], 'profit':[]}
    
    flag = -1

    for i in range(len(data)):
        smadata1 = SMA(data,sma1)[i]
        smadata2 = SMA(data,sma2)[i]
        
        if smadata1 > smadata2:
            if flag !=1:
                summary['buy'].append(data['CLOSE'][i])
                flag =1
        elif smadata1 < smadata2:
            if flag == -1:
                continue
            if flag !=0:
                summary['sell'].append(data['CLOSE'][i])
                flag = 0


    summary['profit']= Diff(summary['sell'], summary['buy'])

    if flag !=0:
        summary['buy'] = summary['buy'][:-1]      

    #ouput = pd.DataFrame(summary)
    #print(ouput)
    
    #print("Winning Sum: {0:.2f} dollar".format(sum(summary['profit'])))

    pos_count = 0
    for num in summary['profit']: 
        # checking condition 
        if num > 0: 
            pos_count += 1
    
        
   
    #print("Winning Percentage: {0:.2f}%".format(pos_count/len(summary['profit']) * 100))
    return "{0:.2f}".format(pos_count/len(summary['profit']) * 100), "{0:.2f}".format(sum(summary['profit'])), len(summary['profit'])

"""
def runthroughcsv(filePath):
        #get all files in the given folder
    fileListing = os.listdir(filePath)
    for myFile in fileListing:
        #create the file path
        myFilePath = os.path.join(filePath, myFile)
        #check to make sure its a file not a sub folder
        if (os.path.isfile(myFilePath) and myFilePath.endswith(".csv")):
            with open(myFilePath, 'r', encoding='utf-8') as csvfile:
                #sniff to find the format
                fileDialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                #create a CSV reader
                myReader = csv.reader(csvfile, dialect=fileDialect)
                #read each row
                for row in myReader:
                    #do your processing here
                    print(row)
                     
               
            with open(myFilePath, 'r', encoding='utf-8') as csvfile: 
                #sniff to find the format 
                fileDialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                #read the CSV file into a dictionary
                dictReader = csv.DictReader(csvfile, dialect=fileDialect)
                for row in dictReader:
                    #do your processing here
                    print(row)    
    return
"""



def runthroughcsv():
    Stocks = {'Stock':["ZS","AYX","ATEYY","LYFT","LOGI","ZEN","NET","DOX","DSCSY","FFIV","GDS","IPGP","PTC","CIEN","ZNGA","AFTPF","WUBA","HII","GSX","AAXN","TKC","JBL","QLYS","CRUS","SLAB","PCCWY","PSN","DXC","BL","MLRYY","CCMP","DSGX","NEWR","JOBS","CYBR","CLGX","ENV","DOYU","PD","TIGO","EPAY","TSEM","MTSI","SAIL","ADS"], 'Gain':[], "Percent":[], "Entry":[]}
    #Stocks = {'Stock':["INGIF"], 'Gain':[], "Percent":[], "Entry":[]}
    for i in range(len(Stocks['Stock'])):
        df = get_quote_data(Stocks['Stock'][i])
        print(Stocks['Stock'][i])
        #print(df)

        gain, percentage, entry = buy_sell_win_long(df, 15, 50)
        Stocks['Percent'].append(gain)
        Stocks['Gain'].append(percentage)
        Stocks['Entry'].append(entry)
        
    #get all files in the given folder
    ouput = pd.DataFrame(Stocks)
    print(ouput)
    print("total entry:{0:.2f}".format( sum(Stocks['Entry'])))
    print("average:{0:.2f}".format( sum(Stocks['Percent'])/len(Stocks['Percent'])))
    

plt.style.use('fivethirtyeight')

"""
datestart = datetime.datetime(2020,1,1)
dateend = datetime.datetime(2020,5,25)
df = web.DataReader("AAPL", data_source='yahoo', interval="1m" , start=datestart, end=dateend)
"""

#df = get_quote_data(sys.argv[1],sys.argv[2],sys.argv[3])
#df = get_quote_data()

#print(buy_sell_win_long(df, 30, 100))

runthroughcsv()

"""
buy_sell = buy_sell(df, 30, 100)
df["buysignal"] = buy_sell[0]
df["sellsignal"] = buy_sell[1]

df.shape


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['CLOSE'], label= 'price', alpha =0.35)
plt.plot(SMA(df,30), label= 'sma30', alpha =0.35)
plt.plot(SMA(df,100), label= 'sma100', alpha =0.35)
plt.scatter(df.index, df['buysignal'], label = 'buy', marker = '^', color = 'green')
plt.scatter(df.index, df['sellsignal'], label = 'sell', marker = 'v', color = 'red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.show()
"""
