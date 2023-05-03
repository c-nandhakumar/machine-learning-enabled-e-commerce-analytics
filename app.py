from flask import Flask,send_file,redirect, url_for,render_template,session
import matplotlib.pyplot as plt
import io
import base64
from flask_wtf import FlaskForm
from wtforms.fields import DateField
from wtforms import validators,SubmitField
from datetime import datetime


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
color_pal = sns.color_palette()

url1 = 'https://drive.google.com/file/d/1siwIxHBMqm9FkcxZv8pXl1yBYNTSXTsL/view?usp=sharing'
file_id2 = url1.split('/')[-2]
dwn_url2 = 'https://drive.google.com/uc?id=' + file_id2

import xgboost as xgb
df =  pd.read_csv(dwn_url2, encoding='ISO-8859-1')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

def plot_creator(x,y,imgname):
    plt.plot(x,y)
    plt.savefig(f'{imgname}.png')

def sales_analysis():
    df.plot(
    linewidth ="0.5",
        figsize=(20, 5),
        color=color_pal[0],
        title='SALES in Rs')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return img_str

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df):
    target_map = df['SALES'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

def predicted_chart(startdate,enddate,df):
    imgname="predicted_sales_chart"
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
    df = df.query('SALES > 19_000').copy()
    df = df.sort_index()
    future = pd.date_range(f'{startdate}',f'{enddate}', freq='1h')
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    df['isFuture'] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)
    future_w_features = df_and_future.query('isFuture').copy()
    reg_new = xgb.XGBRegressor()
    reg_new.load_model('model.json')
    future_w_features['pred'] = reg_new.predict(future_w_features[FEATURES])
    future_w_features['pred'].plot(figsize=(20, 5),
                               color=color_pal[4],
                               ms=1, lw=1,
                               title='Future Predictions')
    img_buffer1 = io.BytesIO()
    plt.savefig(img_buffer1, format='png')
    img_buffer1.seek(0)

    fig, ax2 = plt.subplots(figsize=(20, 5))
    ax2.plot(future_w_features.index, future_w_features['pred'], color='r')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Future Sales')
    ax2.set_title('Predicted Data')
    img_buffer2 = io.BytesIO()
    fig.savefig(img_buffer2, format='png')
    fig.savefig(f'{imgname}.png')
    img_buffer2.seek(0)

    img_str = base64.b64encode(img_buffer1.getvalue()).decode('utf-8')
    img_str1 = base64.b64encode(img_buffer2.getvalue()).decode('utf-8')
    
    # Save the dataframe as an excel file in a particular location
    result_df = future_w_features.copy()
    result_df = result_df.loc[:,['pred']]
    result_df = result_df.reset_index().rename(columns={'index': 'Datetime', 'pred': 'Future Sales'})
    table_html = result_df.to_html(classes="table table-striped")
    file_path = 'data.xlsx'
    result_df.to_excel(file_path, sheet_name='Sheet1')

    return [img_str,img_str1,table_html]



url = 'https://drive.google.com/file/d/1shaEv0kyBHrkmHC0_AF33dyWZRTC0EPL/view?usp=sharing'
file_id = url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?id=' + file_id
data = pd.read_csv(dwn_url, encoding='ISO-8859-1')

df1 = pd.DataFrame(data)

df1 = df1.drop(['STATUS'],axis=1)
df1 = df1.drop(['MSRP'],axis=1)
df1 = df1.drop(['CUSTOMERNAME'],axis=1)
df1 = df1.drop(['PHONE'],axis=1)
df1 = df1.drop(['ADDRESSLINE1'],axis=1)
df1 = df1.drop(['ADDRESSLINE2'],axis=1)
df1 = df1.drop(['CITY'],axis=1)
df1 = df1.drop(['STATE'],axis=1)
df1 = df1.drop(['POSTALCODE'],axis=1)
df1 = df1.drop(['COUNTRY'],axis=1)
df1 = df1.drop(['TERRITORY'],axis=1)
df1 = df1.drop(['CONTACTLASTNAME'],axis=1)
df1 = df1.drop(['CONTACTFIRSTNAME'],axis=1)
df1 = df1.drop(['DEALSIZE'],axis=1)

def generate_bar_plot(data,xlabel):
    fig, ax = plt.subplots(figsize=(5,5))

    plt.hist(data, bins=10, color="#4C72B0")

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image

def generate_count_plot(data,xlabel):
    fig, ax = plt.subplots(figsize=(8,5))
    # Calculate the counts
    counts = data.value_counts()

    # Create the bar plot
    plt.bar(counts.index, counts.values)

    # Define the colors and shapes for each bar
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    # shapes = ["o", "s", "^", "D", "P", "*"]
    # hatch=shapes[i % len(shapes)]
    # Create the bar plot with custom colors and shapes
    for i, (name, count) in enumerate(counts.items()):
        ax.bar(name, count, color=colors[i % len(colors)] )
    plt.xlabel(xlabel)
    plt.ylabel("COUNT")
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image

app = Flask(__name__)
app.config['SECRET_KEY'] = '#$%^&*'
# Bootstrap(app)
# datepicker(app)
class InfoForm(FlaskForm):
    startdate = DateField('Start Date',format="%Y-%m-%d")
    enddate = DateField('End Date',format="%Y-%m-%d")
    submit =SubmitField('Submit')

@app.route('/',methods = ['GET', 'POST'])
def home():
     #price_distribution
    price_distribution = generate_bar_plot(df1["PRICEEACH"],'PRICE')
    #yearly_sales_distribution
    yearly_sales_distribution = generate_count_plot(df1["YEAR_ID"],"YEARS")
    #product_sales_distribution
    product_sales_distribution = generate_count_plot(df1["PRODUCTLINE"],"PRODUCTS")
    form = InfoForm()
    imgurl = sales_analysis()
    if form.validate_on_submit():
        session['startdate'] = form.startdate.data
        session['enddate'] = form.enddate.data
        # session['startdate'] = form.startdate.data
        # session['enddate'] = form.enddate.data
        return redirect(url_for('date'))
    return render_template("index.html",form=form, img=imgurl, pricechart = price_distribution,yearlysales = yearly_sales_distribution,productsales = product_sales_distribution)



@app.route('/date',methods = ['GET','POST'] )
def date():
    print(type(session['startdate']))
   
    startdate_obj = datetime.strptime(session['startdate'], "%a, %d %b %Y %H:%M:%S %Z")
    startdate = startdate_obj.strftime("%Y-%m-%d")
    startdate1 = startdate_obj.strftime("%d/%m/%Y")
    enddate=session['enddate']
    enddate_obj = datetime.strptime(session['enddate'], "%a, %d %b %Y %H:%M:%S %Z")
    enddate1 = enddate_obj.strftime("%d/%m/%Y")
    enddate = enddate_obj.strftime("%Y-%m-%d")

    print(startdate)
    df =  pd.read_csv(dwn_url2, encoding='ISO-8859-1')
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    df = create_features(df)
    
    [imgurl1,imgurl2,table_html]= predicted_chart(startdate,enddate,df)
    return render_template('date.html',startdate=startdate1,enddate=enddate1 ,imgurl1=imgurl1, imgurl2=imgurl2, table= table_html)    


def build_plot():
    img = plot_creator()
    plot_url = base64.b64encode(img.getvalue()).decode()
    return '<img src="data:image/png;base64,{}">'.format(plot_url)


@app.route("/download")
def download_file():
    p = "predicted_sales_chart.png"
    return send_file(p,as_attachment=True)

@app.route("/downloadxl")
def download_xl():
    file_name="data.xlsx"
    return send_file(file_name, as_attachment=True)

if __name__ == '__main__':
    # app.debug = True
    app.run()