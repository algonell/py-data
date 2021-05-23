"""
This is a simple Quantopian algorithm
Incorporates basic ML idea
Uses various stocks

Alpha Idea: 
1. Classify whether the price is going up or down
2. If classification probability is >= threshold then buy
3. If classification probability is < threshold then stay out
"""

import pandas as pd
import talib
import time 

from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

#global parameters
train_data_periods = 504 # train set size
pred_data_periods = 100 # train set size for real time classification
classification_threshold = 0.55 # for signals
vol = 30000 # each entry size in money

def get_data(stock, data, periods):
    fields = ['open', 'high', 'low', 'close', 'volume', 'price']
    train_data = data.history(stock, fields, periods, '1d')    
    
    #keep only true data
    train_data = train_data.dropna()
    
    #feature engineering
    train_data = feature_engineering(train_data)
    
    #keep only true data
    train_data = train_data.dropna()

    #save column names
    cols = list(train_data.columns)
    cols.remove('price')
    
    #make a shift: classify the next price direction
    price_t1 = pd.DataFrame(train_data.values[:-1, -1])#current
    price_t2 = pd.DataFrame(train_data.values[1:, -1])#following

    nominal_class = price_t1 <= price_t2
    nominal_class = nominal_class.astype(int)

    #exclude first row
    train_data = pd.DataFrame(train_data.values[1:, :-1], columns = cols) 
    
    return train_data, nominal_class

def feature_engineering(train_data):
    #SMA
    train_data.insert(0, 'sma20', talib.SMA(train_data['close'].values, timeperiod=20), True)

    #CCI
    train_data.insert(0, 'cci14', talib.CCI(train_data['high'].values, train_data['low'].values, 
                              train_data['close'].values, timeperiod=14), True)

    #RSI
    train_data.insert(0, 'rsi14', talib.RSI(train_data['close'].values, timeperiod=14), True)

    #ADX
    train_data.insert(0, 'adx14', talib.ADX(train_data['high'].values, train_data['low'].values, 
                                            train_data['close'].values, timeperiod=14), True)

    #ATR
    train_data.insert(0, 'atr14', talib.ATR(train_data['high'].values, train_data['low'].values, 
                                            train_data['close'].values, timeperiod=14), True)

    #Bands
    bb20Upperband, bb20Middleband, bb20Lowerband = talib.BBANDS(
        train_data['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    train_data.insert(0, 'bb20Upperband', bb20Upperband, True)
    train_data.insert(0, 'bb20Middleband', bb20Middleband, True)
    train_data.insert(0, 'bb20Lowerband', bb20Lowerband, True)

    bb50Upperband, bb50Middleband, bb50Lowerband = talib.BBANDS(
        train_data['close'].values, timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
    train_data.insert(0, 'bb50Upperband', bb50Upperband, True)
    train_data.insert(0, 'bb50Middleband', bb50Middleband, True)
    train_data.insert(0, 'bb50Lowerband', bb50Lowerband, True)

    #MACD
    macd1226, macdSignal1226, macdHist1226 = talib.MACD(
        train_data['close'].values, fastperiod=13, slowperiod=26, signalperiod=9)
    train_data.insert(0, 'macd1226', macd1226, True)
    train_data.insert(0, 'macdSignal1226', macdSignal1226, True)
    train_data.insert(0, 'macdHist1226', macdHist1226, True)

    #Stochastic
    stochasticSlowK335, stochasticSlowD335 = talib.STOCH(
        train_data['high'].values, train_data['low'].values, train_data['close'].values, 
        fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    train_data.insert(0, 'stochasticSlowK335', stochasticSlowK335, True)
    train_data.insert(0, 'stochasticSlowD335', stochasticSlowD335, True)
    
    return train_data

def create_symbol_list(data):
    """
    Creates list of stocks to be traded
    """          
    
    initial_stock_list = [
        symbol('FB'), 
        symbol('AMZN'), 
        symbol('NDAQ'), 
        symbol('AAPL'), 
        symbol('HD'),
        symbol('V')
    ]
    
    #remove untradable stocks
    return remove_untradable(initial_stock_list, data) 

def remove_untradable(stocks, data):
    to_trade = []
    for stock in stocks:
        if(data.can_trade(stock)):
            to_trade.append(stock)
        else:
            print "can't trade:", stock, "untradable"
            
    return to_trade
    

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
    

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    
    #create symbol list
    context.stocks = create_symbol_list(data)
    
    #train classifiers for each stock
    context.classifiers = {}
    
    for stock in context.stocks:    
        try:
            #get data for training model
            train_data, nominal_class = get_data(stock, data, train_data_periods)     
        
            #fit classifier
            classifier_name = str(stock) + '_classifier'
            context.classifiers[classifier_name] = ensemble.RandomForestClassifier(n_estimators=50, max_depth=10)
            y = nominal_class.values.ravel()
            context.classifiers[classifier_name].fit(train_data.values, y)
        except Exception as e:
            print 'error: before_trading_start', e            
       
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    
    for stock in context.stocks:
        try:
            classifier_name = str(stock) + '_classifier'

            #get data for real-time classification
            train_data, nominal_class = get_data(stock, data, pred_data_periods)   

            #make predictions
            proba = context.classifiers[classifier_name].predict_proba(train_data.values[-1,:].reshape(1, -1))
            #log.info('proba:', proba)

            #check for signals
            buy, sell, out = check_for_signals(proba)

            #trading
            trade(data, context, stock, buy, sell, out)
        except Exception as e:
            print 'error: my_rebalance', e

def check_for_signals(proba):
    """
    Check for trading signals via classification threshold
    """
    
    buy = False
    sell = False
    out = False
    
    if(proba[0][1] >= classification_threshold):   
        buy = True
    else:
        out = True
    
    return buy, sell, out

def trade(data, context, stock, buy, sell, out):
    """
    Enter/exit positions via trading signals
    """
    
    positions = context.portfolio.positions[stock].amount
    amount = vol / int(data.current(stock, 'price'))
    print 'amount:', amount, 'positions:', positions, 'buy:', buy, 'sell:', sell, 'out:', out
    
    #buy signal
    if(buy):
        if(positions <= 0):
            order(stock, amount - positions)
        else:
            order(stock, amount)
    
    #out signal
    if(out):
        if(positions is not 0):
            order(stock, -positions)    
