# -*- coding: utf-8 -*-

#importing the required libraries
from flask import Flask, render_template, request, flash, redirect, url_for, session
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import yfinance as yf
import preprocessor as p
import re
from flask_mysqldb import MySQL
import MySQLdb.cursors
from forms import ContactForm
from flask_mail import Message, Mail
plt.style.use('fivethirtyeight')


import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
mail = Mail()
app = Flask(__name__)


app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USERNAME"] = 'renaissancetechsup@gmail.com'
app.config["MAIL_PASSWORD"] = 'Yg_vy@uib562'

mail.init_app(app)

app.secret_key = 'miniproject'


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Samay@123456'
app.config['MYSQL_DB'] = 'miniproject'


mysql = MySQL(app)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/rennaissancetechnology', methods=['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:

        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE (username = %s OR email = %s) AND password = %s', (username, username, password,))

        account = cursor.fetchone()

        if account:

            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']

            return redirect(url_for('index'))
        else:

            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)


@app.route('/rennaissancetechnology/register', methods=['GET', 'POST'])
def register():
    msg=''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:

        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account1 = cursor.fetchone()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account2 = cursor.fetchone()

        if account1:
            msg = 'Account on this email already exists!'
        elif account2:
            msg = 'Username is taken by any other user! Please choose a different one'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif  (len(username)<5) or (len(username)>10):
            msg = " Username length must be between 5 to 10 characters!"
        elif  (len(password)<6) or (len(password)>18):
            msg = " Password length must be between 6 to 18 characters!"
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:

            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':

        msg = 'Please fill out the form!'

    return render_template('register.html', msg=msg)

@app.route('/rennaissancetechnology/index')
def index():

    if 'loggedin' in session:

        return render_template('index.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/logout')
def logout():

   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)

   return redirect(url_for('login'))

@app.route('/rennaissancetechnology/dossier')
def dossier():

    if 'loggedin' in session:
        return render_template('dossier.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/ourmodel' )
def ourmodel():

    if 'loggedin' in session:

        return render_template('ourmodel.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/statistics')
def statistics():

    if 'loggedin' in session:

        return render_template('statistics.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/contact', methods=['GET', 'POST'])
def contact():

    if 'loggedin' in session:
      form = ContactForm()

      if request.method == 'POST':
        if form.validate() == False:
          flash('All fields are required.')
          return render_template('contact.html', form=form)
        else:
          msg = Message(form.subject.data, sender='renaissancetechsup@gmail.com', recipients=['renaissancetechsup@gmail.com'])
          msg.body = """
          From:-
          Name: %s
          Email: %s
          Message: %s
          """ % (form.name.data, form.email.data, form.message.data)
          mail.send(msg)

          return render_template('contact.html', success=True)

      elif request.method == 'GET':
        return render_template('contact.html', form=form)

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/stockprediction')
def stockprediction():

    if 'loggedin' in session:

        return render_template('stockprediction.html', username=session['username'])

    return redirect(url_for('login'))
@app.route('/rennaissancetechnology/stockprediction-stocks')

def stock():

    if 'loggedin' in session:

        return render_template('stock.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/rennaissancetechnology/stockprediction-fundamental')

def fundamental():

    if 'loggedin' in session:

        return render_template('fundamental.html', username=session['username'])

    return redirect(url_for('login'))
@app.route('/rennaissancetechnology/stockprediction-results',methods = ['POST'])
def insertintotable():
    if 'loggedin' in session:
        nm = request.form['nm']

        #**************** FUNCTIONS TO FETCH DATA ***************************
        def get_historical(quote):
            end = datetime.now()
            start = datetime(end.year-10,end.month,end.day)
            data = yf.download(quote, start=start, end=end)
            df = pd.DataFrame(data=data)
            df.to_csv(''+quote+'.csv')
            if(df.empty):
                ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
                data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')

                data=data.head(503).iloc[::-1]
                data=data.reset_index()

                df=pd.DataFrame()
                df['Date']=data['date']
                df['Open']=data['1. open']
                df['High']=data['2. high']
                df['Low']=data['3. low']
                df['Close']=data['4. close']
                df['Adj Close']=data['5. adjusted close']
                df['Volume']=data['6. volume']
                df.to_csv(''+quote+'.csv',index=False)
            return


        def LSTM_ALGO(df):
            fig1=plt.figure(figsize=(16,8))
            plt.title(quote+' closing price')
            plt.plot(df['Close'])
            plt.xlabel('Days',fontsize=15)
            plt.ylabel('Closing price in INR', fontsize=15)
            plt.savefig('static/Trends.png')
            plt.close(fig1)

            if quote=="^BSESN":
                model = load_model('static/^BSESN_model.h5')
            if quote=="^NSEI":
                model = load_model('static/^NSEI_model.h5')
            if quote=="ASIANPAINT.NS":
                model = load_model('static/ASIANPAINT.NS_model.h5')
            if quote=="BAJFINANCE.NS":
                model = load_model('static/BAJFINANCE.NS_model.h5')
            if quote=="BHARTIARTL.NS":
                model = load_model('static/BHARTIARTL.NS_model.h5')
            if quote=="CIPLA.NS":
                model = load_model('static/CIPLA.NS_model.h5')
            if quote=="ICICIBANK.NS":
                model = load_model('static/ICICIBANK.NS_model.h5')
            if quote=="ITC.NS":
                model = load_model('static/ITC.NS_model.h5')
            if quote=="ONGC.NS":
                model = load_model('static/ONGC.NS_model.h5')
            if quote=="RELIANCE.NS":
                model = load_model('static/RELIANCE.NS_model.h5')
            if quote=="SBIN.NS":
                model = load_model('static/SBIN.NS_model.h5')
            if quote=="TATASTEEL.NS":
                model = load_model('static/TATASTEEL.NS_model.h5')
            if quote=="TCS.NS":
                model = load_model('static/TCS.NS_model.h5')
            if quote=="TECHM.NS":
                model = load_model('static/TECHM.NS_model.h5')

            data = df.filter(['Close'])
            dataset = data.values
            training_data_len= math.ceil(len(dataset)* .8)
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)
            train_data = scaled_data[0:training_data_len , : ]
            x_train = []
            y_train = []
            for i in range(60,len(train_data)):
              x_train.append(train_data[i-60:i, 0])
              y_train.append(train_data[i,0])
            x_train,y_train = np.array(x_train),np.array(y_train)
            x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1))
            test_data= scaled_data[training_data_len-60 : , : ]
            x_test=[]
            y_test=dataset[training_data_len:,:]
            for i in range(60,len(test_data)):
              x_test.append(test_data[i-60:i,0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            rmse = np.sqrt(np.mean(predictions-y_test)**2)
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions']= predictions

            fig2=plt.figure(figsize=(16,8))
            plt.title(quote+' predictions ')
            plt.xlabel('Days', fontsize=15)
            plt.ylabel('closing price (INR)',fontsize=15)
            plt.plot(train['Close'])
            plt.plot(valid[['Close','Predictions']])
            plt.legend(['Training dataset','Actual closing values','Predicted values by our model'],loc='lower left')
            plt.savefig('static/LSTM.png')
            plt.close(fig2)

            end = datetime.now()
            start = datetime(end.year-10,end.month,end.day)
            data = yf.download(quote, start=start, end=end)
            ril_quote = pd.DataFrame(data=data)
            ril_quote.to_csv(''+quote+'.csv')
            new_df = ril_quote.filter(['Close'])
            last_60_days = new_df[-60:].values
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test= []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)
            return pred_price, rmse
        quote=nm

        try:
            get_historical(quote)
        except:
            return render_template('stockprediction.html',not_found=True)
        else:

            #************** PREPROCESSUNG ***********************
            df = pd.read_csv(''+quote+'.csv')
            today_stock=df.iloc[-1:]
            df = df.dropna()
            code_list=[]
            for i in range(0,len(df)):
                code_list.append(quote)
            df2=pd.DataFrame(code_list,columns=['Code'])
            df2 = pd.concat([df2, df], axis=1)
            df=df2

            pred_price, rmse=LSTM_ALGO(df)

            today_stock=today_stock.round(2)
            return render_template('results.html',quote=quote,pred_price=np.round(pred_price,2),open_s=today_stock['Open'].to_string(index=False),
                                   close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                                   high_s=today_stock['High'].to_string(index=False),low_s=today_stock['Low'].to_string(index=False),
                                   vol=today_stock['Volume'].to_string(index=False),rmse=np.round(rmse,2))
        return redirect(url_for('login'))
if __name__ == '__main__':
   app.run()
