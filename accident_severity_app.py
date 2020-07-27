import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

st.markdown("""
## US Accident severity prediction

This is a countrywide traffic accident dataset, which covers 49 states of the United States.
The data is continuously being collected from February 2016 until June 2020, using several data providers, including two APIs which provide streaming traffic event data.
These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras,
and traffic sensors within the road-networks. Currently, there are about 3.0 million accident records in this dataset.
""")

st.markdown("""
This app illustrates the use machine learning to predict the severity or accident based on a set of characteristics.
""")

# Read the dataset:
df = pd.read_csv('df_subset_clean.csv')
#---------------------------------------------

# Which State had the most number of accidents?
data = df['State'].value_counts().head(10)
state_count_acc = pd.value_counts(df['State'])

# Create subplots
fig1 = make_subplots(rows=2, cols=2,
                     specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar", "rowspan":2}],
                           [            None                    ,        None    ]]
                    )
# Add the first trace for US map
fig1.add_trace(go.Choropleth(locations = state_count_acc.index,
                             z = state_count_acc.values.astype(float),
                             locationmode = 'USA-states',
                             colorscale = 'Oranges',
                             colorbar_title = "Count Accidents"),
               row=1, col=1)
# Add the second trace for barchart
fig1.add_trace(go.Bar(x = data.index,
                  text = ['{:,}'.format(val) for val in data.values],
                  textposition = 'auto',
                  textfont = dict(color = '#000000', size=10),
                  y = data.values),
               row=1, col=2)
# Update layout
fig1.update_layout(height=500,
                   showlegend=False,
                   title_text = '2016 - 2020 US Traffic Accident Dataset by State',
                   geo_scope='usa')
# Show plot
st.plotly_chart(fig1, use_container_width=True)
#------------------------------------------

# In which cities did accidents happen the most?
# Get data
accidents_cities = df['City'].value_counts()
accidents_cities = pd.DataFrame(accidents_cities).reset_index().rename(columns={"index": "name", "City": "Accident_counts"})

cities_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
cities_df['name'] = cities_df['name'].apply(lambda x: " ".join(x.split()))

result = cities_df.merge(accidents_cities, how='left', on='name')
result['Accident_counts'].fillna(0, inplace=True)
result = result.drop_duplicates(subset='name')
result = result.sort_values(by=['Accident_counts'], ascending=False).reset_index()

# Plot the city map
result['text'] = result['name'] + '<br>Accident_count ' + (result['Accident_counts']).astype(str)+' cases'
limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
colors = ["crimson","orange","royalblue","lightseagreen","lightgrey"]
cities = []
scale = 1000

fig2 = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = result[lim[0]:lim[1]]
    fig2.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['pop']/scale,
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0} - {1} rank'.format(lim[0],lim[1])))

fig2.update_layout(
        title_text = '2016-2020 US accidents ranking by cities<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

st.plotly_chart(fig2, use_container_width=True)
#-------------------------------------------

st.sidebar.header('User Input Features')
# Collect user input features into DataFrame
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        traffic_signal = st.sidebar.selectbox('Traffic_Signal', (True, False))
        crossing = st.sidebar.selectbox('Crossing', (True, False))
        source = st.sidebar.selectbox('Source', ('MapQuest', 'Bing', 'MapQuest-Bing'))
        side = st.sidebar.selectbox('Side', ('L', 'R'))
        tmc = st.sidebar.slider('TMC', 200, 500, 200)
        stop = st.sidebar.selectbox('Stop', (True, False))
        year = st.sidebar.selectbox('Year', (2016,2017,2018,2019,2020))
        junction = st.sidebar.selectbox('Junction', (True, False))
        sunrise_sunset = st.sidebar.selectbox('Sunrise_Sunset', ('Day', 'Night'))
        station = st.sidebar.selectbox('Station', (True, False))
        data = {'Traffic_Signal': traffic_signal,
                'Crossing': crossing,
                'Source': source,
                'Side': side,
                'TMC': tmc,
                'Stop': stop,
                'Year': year,
                'Junction': junction,
                'Sunrise_Sunset': sunrise_sunset,
                'Station': station
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
#-----------------------------------

# Reduced features
cols = ['Traffic_Signal', 'Crossing', 'Source', 'Side', 'TMC',
        'Stop', 'Year', 'Junction', 'Sunrise_Sunset', 'Station']
df1 = df.copy()
df1 = df1[cols]

# Concat dataset and user input
df1 = pd.concat([input_df, df1], axis=0)

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def label_encoder(list):
  for i in list:
    df1[i] = le.fit_transform(df1[i])
list = ['Source', 'Side', 'Crossing', 'Junction', 'Stop', 'Sunrise_Sunset', 'Traffic_Signal', 'Station']
label_encoder(list=list)

# MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1_scaled= scaler.fit_transform(df1)

df1 = df1[:1] # Select only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df1)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df1)

# Reads in saved classification model
load_clf = pickle.load(open('accident_rfc.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df1)

st.subheader('Prediction')
# accident_severity = np.array(1,2,3,4)
# st.write(accident_severity[prediction])
st.write('The predicted severity level is {}'.format(prediction))
