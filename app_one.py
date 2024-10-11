import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_excel("D://newmoli.xlsx"


df = df[(df['time_difference_minutes'] >= 0) & (df['time_difference_minutes'] <= 5000)]

# Feature engineering
df['job_post_hour'] = df['date_of_job_post'].dt.hour
df['job_post_day'] = df['date_of_job_post'].dt.dayofweek

df['application_hour'] = df['application_date'].dt.hour
df['application_day'] = df['application_date'].dt.dayofweek

label_encoder = LabelEncoder()

df['job_title_encoded'] = label_encoder.fit_transform(df['job_title'])

X = df[[ 'entity_id', 'job_post_day','application_day',  
        'rate', 'certification', 'job_title_encoded']]
y = df['time_difference_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=RandomForestRegressor(n_estimators=150)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

r2=r2_score(y_test,y_pred)

print(r2)

day_mapping = {
    0.0: 'Monday',
    1.0: 'Tuesday',
    2.0: 'Wednesday',
    3.0: 'Thursday',
    4.0: 'Friday',
    5.0: 'Saturday'
}
day_mapping_one = {
    0.0: 'Monday',
    1.0: 'Tuesday',
    2.0: 'Wednesday',
    3.0: 'Thursday',
    4.0: 'Friday',
    5.0: 'Saturday',
    6.0: 'Sunday'
}

df['job_post_day_name']=df['job_post_day'].apply(lambda x:day_mapping[x])
df['application_day_name']=df['application_day'].apply(lambda x:day_mapping_one[x])

# with open('model.pkl', 'wb') as file:
#    pickle.dump(model, file)
    
# Load the model from the file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)



st.title("Time Prediction Model")

rate = st.slider('Select a rate', 55, 80, 65)
languages = ['Select a language'] + df['job_title'].unique().tolist()
selected_language = st.selectbox('Select language:',languages )
cities = ['Select a city'] + df['entity_name'].unique().tolist()
selected_entity = st.selectbox('Select city:', cities)
certifications = ['Select a certification'] + df['certification_name'].unique().tolist()
selected_certification = st.selectbox('Select certification:', certifications)
days = ['Select a day'] + df['job_post_day_name'].unique().tolist()
selected_day = st.selectbox('Select a day:', days)
app_days = ['Select a day'] + df['application_day_name'].unique().tolist()
app_selected_day = st.selectbox('Select a day:', app_days)

if st.button('Calculate Rate'):
    if selected_certification!="Select a certification" and selected_entity!="Select certification" and selected_language!="Select a language" and selected_day !="Select a day" and app_selected_day !="Select a day":
        def find_entity_id(selected_entity):
            return int(df[df['entity_name'] == selected_entity]['entity_id'].unique().item())  
        def find_certification_id(selected_certification):
            return  int(df[df['certification_name'] == selected_certification]['certification'].unique().item())
        def find_job_title_encoded(selected_entity):
            return  int(df[df['job_title'] == selected_entity]['job_title_encoded'].unique().item())
        # def find_job_title_encoded(selected_entity):
        #     return  int(df[df['job_title'] == selected_entity]['job_title_encoded'].unique().item())
        input_data = pd.DataFrame({
        'entity_id': find_entity_id(selected_entity),
        'day1':selected_day,
        'day2':app_selected_day,
        'rate':rate,
        'certification': find_certification_id(selected_certification),
        'job_title_encoded': find_job_title_encoded(selected_language),
        
            },index=[0])
        day_mapping = {
     'Monday':0.0,
     'Tuesday' :1.0,
     'Wednesday':2.0, 
     'Thursday':3.0,
     'Friday':4.0,
     'Saturday' :5.0
}
        app_mapping = {
     'Monday':0.0,
     'Tuesday' :1.0,
     'Wednesday':2.0, 
     'Thursday':3.0,
     'Friday':4.0,
     'Saturday' :5.0,
     'Sunday':6.0
}
        input_data['job_post_day']=input_data['day1'].apply(lambda x: day_mapping[x])
        input_data['application_day']=input_data['day2'].apply(lambda x: app_mapping[x])
        input_data = input_data.drop(columns=['day1', 'day2'])
        # newdf=input_data[[[ 'entity_id', 'job_post_day','application_day',  
        # 'rate', 'certification', 'job_title_encoded']]]
        prediction=loaded_model.predict(input_data[[ 'entity_id', 'job_post_day','application_day',  
        'rate', 'certification', 'job_title_encoded']])
        # st.dataframe(newdf)
        minutes=int(np.ceil(prediction))
        def minutes_to_hours(minutes):
            hours = minutes // 60  # Get hours with floor division
            remaining_minutes = minutes % 60  # Get remaining minutes with modulus
            return hours, remaining_minutes
        hours, minute = minutes_to_hours(minutes)
        st.success(f"The estimated time is {hours} hours and {minute} minutes.")
        st.write(f"Accuracy is {r2}")
        
