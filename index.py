# import libraries
from linecache import cache
from operator import index
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import altair as alt
import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import webbrowser


st.set_page_config(
    page_title="PV Predict", 
    page_icon="🧊", 
    layout="wide",
    initial_sidebar_state="expanded",)

base="light"
font="serif"

st.write("""
# PV power generation prediction using Machine Learning
This graph shows the prediction of PV power generation from the **Recent Power Output** and **Weather Data** for the next 1 hour.
""")

st.sidebar.image('logo.png')
st.sidebar.header('Insert Data')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload the input CSV file", type=["csv"])


# load the input dataset
input_dataset = pd.read_csv(uploaded_file)




st.sidebar.header('Set the sunrise and sunset')
st.cache()
sunrise = st.sidebar.text_input('Sunrise', max_chars=5, value='05:40', help='Example: 05:40')
sunset = st.sidebar.text_input('Sunset', max_chars=5, value='21:00', help='Example: 21:00')



# save the current time
input_dataset['Timestamp'] =  pd.to_datetime(input_dataset['Timestamp'])
current_time = input_dataset['Timestamp'].iloc[0]

# save copy of input dataset
current_data = input_dataset.copy()

# drop the timestamp
input_dataset = input_dataset.drop(['Timestamp'], axis=1)

# separate the weather data for different prediction
input_weather_dataset = input_dataset.copy()
input_weather_dataset = input_weather_dataset.drop(['t-60', 't-55', 't-50', 't-45', 't-40', 't-35', 't-30', 't-25', 't-20', 't-15', 't-10', 't-05'], axis=1)

# import the scalers
scaler_full = pickle.load(open('scaler_full.pkl', 'rb'))
scaler_weather = pickle.load(open('scaler_weather.pkl', 'rb'))

# import the ml model for full dataset
model_full_05 = pickle.load(open('model_full_05.pkl', 'rb'))
model_full_10 = pickle.load(open('model_full_10.pkl', 'rb'))
model_full_15 = pickle.load(open('model_full_15.pkl', 'rb'))
model_full_20 = pickle.load(open('model_full_20.pkl', 'rb'))
model_full_25 = pickle.load(open('model_full_25.pkl', 'rb'))
model_full_30 = pickle.load(open('model_full_30.pkl', 'rb'))
model_full_35 = pickle.load(open('model_full_35.pkl', 'rb'))
model_full_40 = pickle.load(open('model_full_40.pkl', 'rb'))
model_full_45 = pickle.load(open('model_full_45.pkl', 'rb'))
model_full_50 = pickle.load(open('model_full_50.pkl', 'rb'))
model_full_55 = pickle.load(open('model_full_55.pkl', 'rb'))
model_full_60 = pickle.load(open('model_full_60.pkl', 'rb'))

# import the ml model for weather dataset
model_weather_05 = pickle.load(open('model_weather_05.pkl', 'rb'))
model_weather_10 = pickle.load(open('model_weather_10.pkl', 'rb'))
model_weather_15 = pickle.load(open('model_weather_15.pkl', 'rb'))
model_weather_20 = pickle.load(open('model_weather_20.pkl', 'rb'))
model_weather_25 = pickle.load(open('model_weather_25.pkl', 'rb'))
model_weather_30 = pickle.load(open('model_weather_30.pkl', 'rb'))
model_weather_35 = pickle.load(open('model_weather_35.pkl', 'rb'))
model_weather_40 = pickle.load(open('model_weather_40.pkl', 'rb'))
model_weather_45 = pickle.load(open('model_weather_45.pkl', 'rb'))
model_weather_50 = pickle.load(open('model_weather_50.pkl', 'rb'))
model_weather_55 = pickle.load(open('model_weather_55.pkl', 'rb'))
model_weather_60 = pickle.load(open('model_weather_60.pkl', 'rb'))

# scale the input data
input_dataset = scaler_full.transform(input_dataset)
input_weather_dataset = scaler_weather.transform(input_weather_dataset)

# predict output using all features
predict_full_05 = model_full_05.predict(input_dataset)
predict_full_10 = model_full_10.predict(input_dataset)
predict_full_15 = model_full_15.predict(input_dataset)
predict_full_20 = model_full_20.predict(input_dataset)
predict_full_25 = model_full_25.predict(input_dataset)
predict_full_30 = model_full_30.predict(input_dataset)
predict_full_35 = model_full_35.predict(input_dataset)
predict_full_40 = model_full_40.predict(input_dataset)
predict_full_45 = model_full_45.predict(input_dataset)
predict_full_50 = model_full_50.predict(input_dataset)
predict_full_55 = model_full_55.predict(input_dataset)
predict_full_60 = model_full_60.predict(input_dataset)

# predict output using weather data
predict_weather_05 = model_weather_05.predict(input_weather_dataset)
predict_weather_10 = model_weather_10.predict(input_weather_dataset)
predict_weather_15 = model_weather_15.predict(input_weather_dataset)
predict_weather_20 = model_weather_20.predict(input_weather_dataset)
predict_weather_25 = model_weather_25.predict(input_weather_dataset)
predict_weather_30 = model_weather_30.predict(input_weather_dataset)
predict_weather_35 = model_weather_35.predict(input_weather_dataset)
predict_weather_40 = model_weather_40.predict(input_weather_dataset)
predict_weather_45 = model_weather_45.predict(input_weather_dataset)
predict_weather_50 = model_weather_50.predict(input_weather_dataset)
predict_weather_55 = model_weather_55.predict(input_weather_dataset)
predict_weather_60 = model_weather_60.predict(input_weather_dataset)



# to make sure there is no negative value in prediction
if predict_full_05 < 0:
    predict_full_05 = np.array([0])
else:
    predict_full_05 = predict_full_05

if predict_full_10 < 0:
    predict_full_10 = np.array([0])
else:
    predict_full_10 = predict_full_10

if predict_full_15 < 0:
    predict_full_15 = np.array([0])
else:
    predict_full_15 = predict_full_15

if predict_full_20 < 0:
    predict_full_20 = np.array([0])
else:
    predict_full_20 = predict_full_20

if predict_full_25 < 0:
    predict_full_25 = np.array([0])
else:
    predict_full_25 = predict_full_25

if predict_full_30 < 0:
    predict_full_30 = np.array([0])
else:
    predict_full_30 = predict_full_30

if predict_full_35 < 0:
    predict_full_35 = np.array([0])
else:
    predict_full_35 = predict_full_35

if predict_full_40 < 0:
    predict_full_40 = np.array([0])
else:
    predict_full_40 = predict_full_40

if predict_full_45 < 0:
    predict_full_45 = np.array([0])
else:
    predict_full_45 = predict_full_45

if predict_full_50 < 0:
    predict_full_50 = np.array([0])
else:
    predict_full_50 = predict_full_50

if predict_full_55 < 0:
    predict_full_55 = np.array([0])
else:
    predict_full_55 = predict_full_55

if predict_full_60 < 0:
    predict_full_60 = np.array([0])
else:
    predict_full_60 = predict_full_60


if predict_weather_05 < 0:
    predict_weather_05 = np.array([0])
else:
    predict_weather_05 = predict_weather_05

if predict_weather_10 < 0:
    predict_weather_10 = np.array([0])
else:
    predict_weather_10 = predict_weather_10

if predict_weather_15 < 0:
    predict_weather_15 = np.array([0])
else:
    predict_weather_15 = predict_weather_15

if predict_weather_20 < 0:
    predict_weather_20 = np.array([0])
else:
    predict_weather_20 = predict_weather_20

if predict_weather_25 < 0:
    predict_weather_25 = np.array([0])
else:
    predict_weather_25 = predict_weather_25

if predict_weather_30 < 0:
    predict_weather_30 = np.array([0])
else:
    predict_weather_30 = predict_weather_30

if predict_weather_35 < 0:
    predict_weather_35 = np.array([0])
else:
    predict_weather_35 = predict_weather_35

if predict_weather_40 < 0:
    predict_weather_40 = np.array([0])
else:
    predict_weather_40 = predict_weather_40

if predict_weather_45 < 0:
    predict_weather_45 = np.array([0])
else:
    predict_weather_45 = predict_weather_45

if predict_weather_50 < 0:
    predict_weather_50 = np.array([0])
else:
    predict_weather_50 = predict_weather_50

if predict_weather_55 < 0:
    predict_weather_55 = np.array([0])
else:
    predict_weather_55 = predict_weather_55

if predict_weather_60 < 0:
    predict_weather_60 = np.array([0])
else:
    predict_weather_60 = predict_weather_60


# to make sure no power output before sunrise or after sunset

sunrise = datetime.datetime.strptime(sunrise, "%H:%M").time()
sunset = datetime.datetime.strptime(sunset, "%H:%M").time()

if (current_time + pd.Timedelta(minutes=5)).time() < sunrise:
    predict_full_05 = np.array([0])
elif (current_time + pd.Timedelta(minutes=5)).time() > sunset:
    predict_full_05 = np.array([0])
else:
    predict_full_05 = predict_full_05

if (current_time + pd.Timedelta(minutes=10)).time() < sunrise:
    predict_full_10 = np.array([0])
elif (current_time + pd.Timedelta(minutes=10)).time() > sunset:
    predict_full_10 = np.array([0])
else:
    predict_full_10 = predict_full_10

if (current_time + pd.Timedelta(minutes=15)).time() < sunrise:
    predict_full_15 = np.array([0])
elif (current_time + pd.Timedelta(minutes=15)).time() > sunset:
    predict_full_15 = np.array([0])
else:
    predict_full_15 = predict_full_15

if (current_time + pd.Timedelta(minutes=20)).time() < sunrise:
    predict_full_20 = np.array([0])
elif (current_time + pd.Timedelta(minutes=20)).time() > sunset:
    predict_full_20 = np.array([0])
else:
    predict_full_20 = predict_full_20

if (current_time + pd.Timedelta(minutes=25)).time() < sunrise:
    predict_full_25 = np.array([0])
elif (current_time + pd.Timedelta(minutes=25)).time() > sunset:
    predict_full_25 = np.array([0])
else:
    predict_full_25 = predict_full_25


if (current_time + pd.Timedelta(minutes=30)).time() < sunrise:
    predict_full_30 = np.array([0])
elif (current_time + pd.Timedelta(minutes=30)).time() > sunset:
    predict_full_30 = np.array([0])
else:
    predict_full_30 = predict_full_30


if (current_time + pd.Timedelta(minutes=35)).time() < sunrise:
    predict_full_35 = np.array([0])
elif (current_time + pd.Timedelta(minutes=35)).time() > sunset:
    predict_full_35 = np.array([0])
else:
    predict_full_35 = predict_full_35


if (current_time + pd.Timedelta(minutes=40)).time() < sunrise:
    predict_full_40 = np.array([0])
elif (current_time + pd.Timedelta(minutes=40)).time() > sunset:
    predict_full_40 = np.array([0])
else:
    predict_full_40 = predict_full_40


if (current_time + pd.Timedelta(minutes=45)).time() < sunrise:
    predict_full_45 = np.array([0])
elif (current_time + pd.Timedelta(minutes=45)).time() > sunset:
    predict_full_45 = np.array([0])
else:
    predict_full_45 = predict_full_45


if (current_time + pd.Timedelta(minutes=50)).time() < sunrise:
    predict_full_50 = np.array([0])
elif (current_time + pd.Timedelta(minutes=50)).time() > sunset:
    predict_full_50 = np.array([0])
else:
    predict_full_50 = predict_full_50


if (current_time + pd.Timedelta(minutes=55)).time() < sunrise:
    predict_full_55 = np.array([0])
elif (current_time + pd.Timedelta(minutes=55)).time() > sunset:
    predict_full_55 = np.array([0])
else:
    predict_full_55 = predict_full_55


if (current_time + pd.Timedelta(minutes=60)).time() < sunrise:
    predict_full_60 = np.array([0])
elif (current_time + pd.Timedelta(minutes=60)).time() > sunset:
    predict_full_60 = np.array([0])
else:
    predict_full_60 = predict_full_60


if (current_time + pd.Timedelta(minutes=5)).time() < sunrise:
    predict_weather_05 = np.array([0])
elif (current_time + pd.Timedelta(minutes=5)).time() > sunset:
    predict_weather_05 = np.array([0])
else:
    predict_weather_05 = predict_weather_05

if (current_time + pd.Timedelta(minutes=10)).time() < sunrise:
    predict_weather_10 = np.array([0])
elif (current_time + pd.Timedelta(minutes=10)).time() > sunset:
    predict_weather_10 = np.array([0])
else:
    predict_weather_10 = predict_weather_10

if (current_time + pd.Timedelta(minutes=15)).time() < sunrise:
    predict_weather_15 = np.array([0])
elif (current_time + pd.Timedelta(minutes=15)).time() > sunset:
    predict_weather_15 = np.array([0])
else:
    predict_weather_15 = predict_weather_15

if (current_time + pd.Timedelta(minutes=20)).time() < sunrise:
    predict_weather_20 = np.array([0])
elif (current_time + pd.Timedelta(minutes=20)).time() > sunset:
    predict_weather_20 = np.array([0])
else:
    predict_weather_20 = predict_weather_20

if (current_time + pd.Timedelta(minutes=25)).time() < sunrise:
    predict_weather_25 = np.array([0])
elif (current_time + pd.Timedelta(minutes=25)).time() > sunset:
    predict_weather_25 = np.array([0])
else:
    predict_weather_25 = predict_weather_25


if (current_time + pd.Timedelta(minutes=30)).time() < sunrise:
    predict_weather_30 = np.array([0])
elif (current_time + pd.Timedelta(minutes=30)).time() > sunset:
    predict_weather_30 = np.array([0])
else:
    predict_weather_30 = predict_weather_30


if (current_time + pd.Timedelta(minutes=35)).time() < sunrise:
    predict_weather_35 = np.array([0])
elif (current_time + pd.Timedelta(minutes=35)).time() > sunset:
    predict_weather_35 = np.array([0])
else:
    predict_weather_35 = predict_weather_35


if (current_time + pd.Timedelta(minutes=40)).time() < sunrise:
    predict_weather_40 = np.array([0])
elif (current_time + pd.Timedelta(minutes=40)).time() > sunset:
    predict_weather_40 = np.array([0])
else:
    predict_weather_40 = predict_weather_40


if (current_time + pd.Timedelta(minutes=45)).time() < sunrise:
    predict_weather_45 = np.array([0])
elif (current_time + pd.Timedelta(minutes=45)).time() > sunset:
    predict_weather_45 = np.array([0])
else:
    predict_weather_45 = predict_weather_45


if (current_time + pd.Timedelta(minutes=50)).time() < sunrise:
    predict_weather_50 = np.array([0])
elif (current_time + pd.Timedelta(minutes=50)).time() > sunset:
    predict_weather_50 = np.array([0])
else:
    predict_weather_50 = predict_weather_50


if (current_time + pd.Timedelta(minutes=55)).time() < sunrise:
    predict_weather_55 = np.array([0])
elif (current_time + pd.Timedelta(minutes=55)).time() > sunset:
    predict_weather_55 = np.array([0])
else:
    predict_weather_55 = predict_weather_55


if (current_time + pd.Timedelta(minutes=60)).time() < sunrise:
    predict_weather_60 = np.array([0])
elif (current_time + pd.Timedelta(minutes=60)).time() > sunset:
    predict_weather_60 = np.array([0])
else:
    predict_weather_60 = predict_weather_60




predicted_data = pd.DataFrame([
    [current_time - pd.Timedelta(minutes=55), 'Current Output', current_data['t-60'].iloc[0]],
    [current_time - pd.Timedelta(minutes=50), 'Current Output', current_data['t-55'].iloc[0]],
    [current_time - pd.Timedelta(minutes=45), 'Current Output', current_data['t-50'].iloc[0]],
    [current_time - pd.Timedelta(minutes=40), 'Current Output', current_data['t-45'].iloc[0]],
    [current_time - pd.Timedelta(minutes=35), 'Current Output', current_data['t-40'].iloc[0]],
    [current_time - pd.Timedelta(minutes=30), 'Current Output', current_data['t-35'].iloc[0]],
    [current_time - pd.Timedelta(minutes=25), 'Current Output', current_data['t-30'].iloc[0]],
    [current_time - pd.Timedelta(minutes=20), 'Current Output', current_data['t-25'].iloc[0]],
    [current_time - pd.Timedelta(minutes=15), 'Current Output', current_data['t-20'].iloc[0]],
    [current_time - pd.Timedelta(minutes=10), 'Current Output', current_data['t-15'].iloc[0]],
    [current_time - pd.Timedelta(minutes=5), 'Current Output', current_data['t-10'].iloc[0]],
    [current_time - pd.Timedelta(minutes=0), 'Current Output', current_data['t-05'].iloc[0]],
    [current_time - pd.Timedelta(minutes=0), 'Predicted Output', current_data['t-05'].iloc[0]],
    [current_time + pd.Timedelta(minutes=5), 'Predicted Output', predict_full_05.round(1).item()],
    [current_time + pd.Timedelta(minutes=10), 'Predicted Output', predict_full_10.round(1).item()],
    [current_time + pd.Timedelta(minutes=15), 'Predicted Output', predict_full_15.round(1).item()],
    [current_time + pd.Timedelta(minutes=20), 'Predicted Output', predict_full_20.round(1).item()],
    [current_time + pd.Timedelta(minutes=25), 'Predicted Output', predict_full_25.round(1).item()],
    [current_time + pd.Timedelta(minutes=30), 'Predicted Output', predict_full_30.round(1).item()],
    [current_time + pd.Timedelta(minutes=35), 'Predicted Output', predict_full_35.round(1).item()],
    [current_time + pd.Timedelta(minutes=40), 'Predicted Output', predict_full_40.round(1).item()],
    [current_time + pd.Timedelta(minutes=45), 'Predicted Output', predict_full_45.round(1).item()],
    [current_time + pd.Timedelta(minutes=50), 'Predicted Output', predict_full_50.round(1).item()],
    [current_time + pd.Timedelta(minutes=55), 'Predicted Output', predict_full_55.round(1).item()],
    [current_time + pd.Timedelta(minutes=60), 'Predicted Output', predict_full_60.round(1).item()],
    [current_time - pd.Timedelta(minutes=0), 'Alternative Prediction', current_data['t-05'].iloc[0]],
    [current_time + pd.Timedelta(minutes=5), 'Alternative Prediction', predict_weather_05.round(1).item()],
    [current_time + pd.Timedelta(minutes=10), 'Alternative Prediction', predict_weather_10.round(1).item()],
    [current_time + pd.Timedelta(minutes=15), 'Alternative Prediction', predict_weather_15.round(1).item()],
    [current_time + pd.Timedelta(minutes=20), 'Alternative Prediction', predict_weather_20.round(1).item()],
    [current_time + pd.Timedelta(minutes=25), 'Alternative Prediction', predict_weather_25.round(1).item()],
    [current_time + pd.Timedelta(minutes=30), 'Alternative Prediction', predict_weather_30.round(1).item()],
    [current_time + pd.Timedelta(minutes=35), 'Alternative Prediction', predict_weather_35.round(1).item()],
    [current_time + pd.Timedelta(minutes=40), 'Alternative Prediction', predict_weather_40.round(1).item()],
    [current_time + pd.Timedelta(minutes=45), 'Alternative Prediction', predict_weather_45.round(1).item()],
    [current_time + pd.Timedelta(minutes=50), 'Alternative Prediction', predict_weather_50.round(1).item()],
    [current_time + pd.Timedelta(minutes=55), 'Alternative Prediction', predict_weather_55.round(1).item()],
    [current_time + pd.Timedelta(minutes=60), 'Alternative Prediction', predict_weather_60.round(1).item()]], 
    columns=['time', 'type', 'power'])

predicted_data['time'] = pd.to_datetime(predicted_data['time'])
predicted_data['time'] = predicted_data['time'].dt.strftime('%H:%M')



prediction_chart = pd.DataFrame([
    [current_time + pd.Timedelta(minutes=5), predict_full_05.round(1).item(), predict_weather_05.round(1).item()],
    [current_time + pd.Timedelta(minutes=10), predict_full_10.round(1).item(), predict_weather_10.round(1).item()],
    [current_time + pd.Timedelta(minutes=15), predict_full_15.round(1).item(), predict_weather_15.round(1).item()],
    [current_time + pd.Timedelta(minutes=20), predict_full_20.round(1).item(), predict_weather_20.round(1).item()],
    [current_time + pd.Timedelta(minutes=25), predict_full_25.round(1).item(), predict_weather_25.round(1).item()],
    [current_time + pd.Timedelta(minutes=30), predict_full_30.round(1).item(), predict_weather_30.round(1).item()],
    [current_time + pd.Timedelta(minutes=35), predict_full_35.round(1).item(), predict_weather_35.round(1).item()],
    [current_time + pd.Timedelta(minutes=40), predict_full_40.round(1).item(), predict_weather_40.round(1).item()],
    [current_time + pd.Timedelta(minutes=45), predict_full_45.round(1).item(), predict_weather_45.round(1).item()],
    [current_time + pd.Timedelta(minutes=50), predict_full_50.round(1).item(), predict_weather_50.round(1).item()],
    [current_time + pd.Timedelta(minutes=55), predict_full_55.round(1).item(), predict_weather_55.round(1).item()],
    [current_time + pd.Timedelta(minutes=60), predict_full_60.round(1).item(), predict_weather_60.round(1).item()]],
    columns=['Time', 'Predicted PV Power Output', 'Alternative Prediction by Weather'])


prediction_chart['Time'] = pd.to_datetime(prediction_chart['Time'])
prediction_chart['Time'] = prediction_chart['Time'].dt.strftime('%H:%M')

prediction_chart = prediction_chart.set_index('Time')




# altair chart

source2 = predicted_data.copy()


nearest = alt.selection_single(nearest=True, on='mouseover', fields=['time'], empty='none', clear="mouseout")

domain = ['Current Output','Predicted Output', 'Alternative Prediction']
range_ = ['blue', 'red', 'darkgrey']

line = alt.Chart(source2).mark_line(interpolate='basis').encode(
    x= alt.X('time', axis=alt.Axis(labelAngle= 0, title='5 Minute Intervals')),
    y= alt.Y('power', axis=alt.Axis(title='Power Output')),
    color=alt.Color('type', legend=alt.Legend(title="Prediction Type"), scale=alt.Scale(domain=domain, range=range_)))

selectors = alt.Chart(source2).mark_point().encode(x='time', opacity=alt.value(0)).add_selection(nearest)

points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'power', alt.value('')))

rules = alt.Chart(source2).mark_rule(color='gray').encode(x ='time').transform_filter(nearest)

predict_chart = alt.layer(line, selectors, points, rules, text).properties(width=1200, height=400).configure_axis(grid=True)

st.write(predict_chart)




st.subheader('**Table:** PV power generation prediction for the next 1 hour in five-minute intervals')


st.table(prediction_chart.style.format({"Predicted PV Power Output": "{:.2f}", "Alternative Prediction by Weather": "{:.2f}"},))

url = 'https://raw.githubusercontent.com/gazihasanrahman/Predicting-PV-power-generation-using-ML/Prediction.csv'
# url = 'file:///C:/Users/hasan/OneDrive%20-%20University%20of%20East%20Anglia/Dissertation%20(7027X)/PV%20power%20using%20ML/webapp/Prediction.csv'

if st.sidebar.button('Export your result'):
    webbrowser.open_new_tab(url)



