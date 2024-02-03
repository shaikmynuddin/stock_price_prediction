import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import plotly.graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = 2
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line_color='white'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line_color='green'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


# Data Preprocessing

st.subheader('Data Preprocessing')
st.text("---------------------------------------------------------------------------")
st.text(" Data Preprocessing ")
st.text("---------------------------------------------------------------------------")
st.text(data.isnull().sum())
st.text("---------------------------------------------------------------------------")

#FEAture selection corelation 

st.subheader('Correlation')
corr = data[['Close', 'High']].corr().iloc[0, 1]
st.text('Pearsons correlation: {:.2f}'.format(corr))


# Data Splitting``

st.subheader('Data Splitting')
X_train, X_test, Y_train, Y_test = train_test_split(data[['Open', 'High', 'Low']], data['Close'], test_size=0.3, random_state=40)
st.text('Total number of data in input: {}'.format(len(data)))
st.text('Total number of data in training part: {}'.format(len(X_train)))
st.text('Total number of data in testing part: {}'.format(len(X_test)))

# Support Vector Regression

svt_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svt_rbf.fit(X_train, Y_train)
y_rbf = svt_rbf.predict(X_test)

# Lasso Regression
reg_lasso = linear_model.Lasso()
reg_lasso.fit(X_train, Y_train)
prd_lasso = reg_lasso.predict(X_test)

# Evaluation Metrics
st.subheader('Evaluation Metrics')

# Support Vector Regression
st.text("---------------------------------------------------------------------------")
st.text(" Support Vector Regression")
st.text("---------------------------------------------------------------------------")
score_svr_mae = metrics.mean_absolute_error(Y_test, y_rbf)
score_svr_mse = metrics.mean_squared_error(Y_test, y_rbf)
score_svr_rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_rbf))
st.text(f'1. Mean Absolute Error: {score_svr_mae}')
st.text(f'2. Mean Squared Error: {score_svr_mse}')
st.text(f'3. Root Mean Squared Error: {score_svr_rmse}')
st.text("---------------------------------------------------------------------------")

# Lasso Regression
st.text("---------------------------------------------------------------------------")
st.text(" Lasso Regression ")
st.text("---------------------------------------------------------------------------")
score_lasso_mae = metrics.mean_absolute_error(Y_test, prd_lasso)
score_lasso_mse = metrics.mean_squared_error(Y_test, prd_lasso)
score_lasso_rmse = np.sqrt(metrics.mean_squared_error(Y_test, prd_lasso))
st.text(f'1. Mean Absolute Error: {score_lasso_mae}')
st.text(f'2. Mean Squared Error: {score_lasso_mse}')
st.text(f'3. Root Mean Squared Error: {score_lasso_rmse}')
st.text("---------------------------------------------------------------------------")

# Prediction
st.subheader('Prediction')
st.text("---------------------------------------------------------------------------")
st.text(" Prediction ")
st.text("---------------------------------------------------------------------------")

# Define future_dates
future_dates = pd.date_range(data['Date'].iloc[-1] + timedelta(days=1), periods=10, freq='D')

present_date = pd.Timestamp(date.today())  # Convert datetime.date to Pandas Timestamp

for i in range(10):
    if future_dates[i] > present_date:
        st.text("---------------------------------------------------------------------------")
        st.text(f"Closed Price on {future_dates[i].strftime('%d-%b-%Y')}: {prd_lasso[i]}")
        st.text(f"   Stock Price Date= {future_dates[i].strftime('%d-%b-%Y')}")
        st.text("---------------------------------------------------------------------------")


# Prediction of Stock
st.subheader('Prediction of Stock')
fig_prediction = go.Figure()
fig_prediction.add_trace(go.Scatter(x=future_dates, y=prd_lasso, mode='lines', name='Lasso Regression Prediction'))
fig_prediction.update_layout(title_text='Prediction of Stock', xaxis_title='Date', yaxis_title='Predicted Price')
st.plotly_chart(fig_prediction)

# Comparison Graph -- Error Values (using Plotly)
st.subheader('Comparison Graph -- Error Values')
fig_comparison = go.Figure()

# Data for the bar chart
methods = ['SVR', 'Lasso Regression']
mae_scores = [score_svr_mae, score_lasso_mae]
mse_scores = [score_svr_mse, score_lasso_mse]
rmse_scores = [score_svr_rmse, score_lasso_rmse]

# Adding traces for MAE, MSE, RMSE
fig_comparison.add_trace(go.Bar(x=methods, y=mae_scores, name='Mean Absolute Error (MAE)'))
fig_comparison.add_trace(go.Bar(x=methods, y=mse_scores, name='Mean Squared Error (MSE)'))
fig_comparison.add_trace(go.Bar(x=methods, y=rmse_scores, name='Root Mean Squared Error (RMSE)'))

# Updating layout
fig_comparison.update_layout(barmode='group', title='Comparison Graph -- Error Values', xaxis_title='Method', yaxis_title='Performance')
st.plotly_chart(fig_comparison)