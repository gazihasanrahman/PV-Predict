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
# input_dataset = pd.read_csv('input_1.csv')
input_dataset = pd.read_csv(uploaded_file)




st.sidebar.header('Set the sunrise and sunset')
st.cache()
sunrise = st.sidebar.text_input('Sunrise', max_chars=5, value='05:40', help='Example: 05:40')
sunset = st.sidebar.text_input('Sunset', max_chars=5, value='21:00', help='Example: 21:00')







# if uploaded_file is not None:
#     input_dataset = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
#         sex = st.sidebar.selectbox('Sex',('male','female'))
#         bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
#         bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
#         flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
#         body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
#         data = {'island': island,
#                 'bill_length_mm': bill_length_mm,
#                 'bill_depth_mm': bill_depth_mm,
#                 'flipper_length_mm': flipper_length_mm,
#                 'body_mass_g': body_mass_g,
#                 'sex': sex}
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_dataset = user_input_features()

# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_dataset,penguins],axis=0)




























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





# if uploaded_file is not None:
#     input_dataset = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
#         sex = st.sidebar.selectbox('Sex',('male','female'))
#         bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
#         bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
#         flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
#         body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
#         data = {'island': island,
#                 'bill_length_mm': bill_length_mm,
#                 'bill_depth_mm': bill_depth_mm,
#                 'flipper_length_mm': flipper_length_mm,
#                 'body_mass_g': body_mass_g,
#                 'sex': sex}
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_dataset = user_input_features()

# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_dataset,penguins],axis=0)


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




predicted_data.to_csv('predicted_data.csv')



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




# input_dataset = input_dataset.sort_values(by=['Timestamp'], ascending=True)



# # set dawn and dusk
# dawn = 285
# dusk = 1120

# # copy the input data to X_test and pv_prediction
# X_test = input_dataset.copy()
# pv_prediction = input_dataset.copy()

# # separate the Month, Day and Time
# X_test['Month'] = pd.DatetimeIndex(X_test['Timestamp']).month
# X_test['Day'] = pd.DatetimeIndex(X_test['Timestamp']).day
# X_test['Time'] = X_test['Timestamp'].dt.hour*60 + X_test['Timestamp'].dt.minute

# # keep only last hour's record
# X_test = X_test.sort_values(by=['Timestamp'], ascending=True)
# X_test = X_test.head(12)
# start_time = X_test['Timestamp'].iloc[0]
# end_time = X_test['Timestamp'].iloc[-1]

# # identify for which time prediction is going to be made
# M5 = end_time + pd.Timedelta(minutes=5)
# M10 = end_time + pd.Timedelta(minutes=10)
# M15 = end_time + pd.Timedelta(minutes=15)
# M20 = end_time + pd.Timedelta(minutes=20)
# M25 = end_time + pd.Timedelta(minutes=25)
# M30 = end_time + pd.Timedelta(minutes=30)
# M35 = end_time + pd.Timedelta(minutes=35)
# M40 = end_time + pd.Timedelta(minutes=40)
# M45 = end_time + pd.Timedelta(minutes=45)
# M50 = end_time + pd.Timedelta(minutes=50)
# M55 = end_time + pd.Timedelta(minutes=55)
# M60 = end_time + pd.Timedelta(minutes=60)

# # reset index
# X_test = X_test.reset_index()
# X_test = X_test.drop(['index'], axis=1)

# X_test = X_test.transpose()

# X_test = X_test.drop(X_test.index[[0,2,3,4]])
# X_test.reset_index(drop=True, inplace=True)

# #add time to the dataframe
# X_test['Timestamp'] = M5
# X_test['Month'] = pd.DatetimeIndex(X_test['Timestamp']).month
# X_test['Day'] = pd.DatetimeIndex(X_test['Timestamp']).day
# X_test['Time'] = X_test['Timestamp'].dt.hour*60 + X_test['Timestamp'].dt.minute
# X_test = X_test.drop(['Timestamp'], axis=1)

# # rearrange and change column name
# cols = X_test.columns.tolist()
# cols = cols[-3:] + cols[:-3]
# X_test = X_test[cols]
# X_test.columns = ['Month', 'Day', 'Time', 't-60', 't-55', 't-50', 't-45', 't-40', 't-35', 't-30', 't-25', 't-20', 't-15', 't-10', 't-5']

# # import the scaler
# scaling = pickle.load(open('scaler.pkl', 'rb'))

# # scale the imported data using standard scaler
# X_test = scaling.transform(X_test)

# # import the ml model
# model = pickle.load(open('ml_model.pkl', 'rb'))

# # make the prediction
# y_pred = model.predict(X_test)

# # to make sure there is no negative value in prediction
# if y_pred < 0:
#     y_pred = 0
# else:
#     y_pred = y_pred

# # to make sure no power output before dawn or after dusk
# if (M5.hour*60 + M5.minute) < dawn:
#     y_pred = 0
# elif (M5.hour*60 + M5.minute) > dusk:
#     y_pred = 0
# else:
#     y_pred = y_pred

# # add the prodiction to the input data
# pv_prediction.loc[-1] = [M5, y_pred.round(1).item()]
# pv_prediction.reset_index(drop=True, inplace=True)

# pv_prediction.to_csv('pv_prediction.csv', index=None)















# altair chart

source2 = predicted_data.copy()
# source2 = pd.read_csv("pv_prediction2.csv")


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

# alt.layer(line, selectors, points, rules, text).properties(width=800, height=400).configure_axis(grid=True)

predict_chart = alt.layer(line, selectors, points, rules, text).properties(width=1200, height=400).configure_axis(grid=True)

st.write(predict_chart)




st.subheader('**Table:** PV power generation prediction for the next 1 hour in five-minute intervals')

# st.table(data=prediction_chart.round(2))

st.table(prediction_chart.style.format({"Predicted PV Power Output": "{:.2f}", "Alternative Prediction by Weather": "{:.2f}"},))


url = 'file:///C:/Users/hasan/OneDrive%20-%20University%20of%20East%20Anglia/Dissertation%20(7027X)/PV%20power%20using%20ML/webapp/Prediction.csv'

if st.sidebar.button('Export your result'):
    webbrowser.open_new_tab(url)



