from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import streamlit as st
import pickle
 
# Load data
df_interpreter = pd.read_excel('interpreter.xlsx')
df_job_statistics = pd.read_excel('job_statistics.xlsx')
 
# Preprocess data
grouped_df = df_interpreter.groupby("language_name").agg(
    interpreter_name=("interpreter_name", lambda x: ", ".join(x)),
    uncertified_interpreter_count=("uncertified_interpreter_count", "sum"),
    aoc_certified_interpreter_count=("aoc_certified_interpreter_count", "sum"),
    aoc_registered_interpreter_count=("aoc_registered_interpreter_count", "sum")
)
df_interpreter = grouped_df.reset_index()
df_interpreter = df_interpreter.drop('interpreter_name', axis=1)
df_interpreter = df_interpreter.rename(columns={'language_name': 'job_title'})
df_interpreter = df_interpreter.rename(columns={
    'uncertified_interpreter_count': 'certificate_type_1',
    'aoc_certified_interpreter_count': 'certificate_type_2',
    'aoc_registered_interpreter_count': 'certificate_type_3'
})
df = pd.merge(df_job_statistics, df_interpreter, on='job_title', how='left')
df['date_of_job_post'] = pd.to_datetime(df['date_of_job_post'], format='%d-%m-%Y %H:%M:%S')
df['job_post_hour'] = df['date_of_job_post'].dt.hour
df['job_post_minutes'] = df['date_of_job_post'].dt.minute
df['job_post_day'] = df['date_of_job_post'].dt.dayofweek
df.drop('date_of_job_post', axis=1, inplace=True)
 
# Separate data by certificate type
certification_1 = df[df['certificate'] == 1]
certification_2 = df[df['certificate'] == 2]
certification_3 = df[df['certificate'] == 3]
 
# Drop irrelevant columns
columns_to_drop = ['job_id', 'entity_id', 'modality_id', 'interpreter_id',
                   'time_difference_hours', 'interpreter_name',
                   'contact_email', 'rate', 'certificate_name', 'application_status', 'application_status_name',
                   'accepted_date', 'certificate_type_2', 'certificate_type_3']
certification_1 = certification_1.drop(columns=columns_to_drop, errors='ignore')
columns_to_drop = ['job_id', 'entity_id', 'modality_id', 'interpreter_id',
                   'time_difference_hours', 'interpreter_name',
                   'contact_email', 'rate', 'certificate_name', 'application_status', 'application_status_name',
                   'accepted_date', 'certificate_type_1', 'certificate_type_3']
certification_2 = certification_2.drop(columns=columns_to_drop, errors='ignore')
columns_to_drop = ['job_id', 'entity_id', 'modality_id', 'interpreter_id',
                   'time_difference_hours', 'interpreter_name',
                   'contact_email', 'rate', 'certificate_name', 'application_status', 'application_status_name',
                   'accepted_date', 'certificate_type_1', 'certificate_type_2']
certification_3 = certification_3.drop(columns=columns_to_drop, errors='ignore')
 
# Rename interpreter availability columns
certification_1 = certification_1.rename(columns={'certificate_type_1': 'interpreter_available'})
certification_2 = certification_2.rename(columns={'certificate_type_2': 'interpreter_available'})
certification_3 = certification_3.rename(columns={'certificate_type_3': 'interpreter_available'})
 
# Merge dataframes
df_merged = pd.concat([certification_1, certification_2, certification_3], axis=0)
 
df_merged = df_merged[(df_merged['time_difference_minutes'] >= 58) & (df_merged['time_difference_minutes'] <= 1500)]
 
df_merged = df_merged.drop('interpreter_available', axis=1)
df_mergerr = df_merged.copy()
 
 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
 
# Encode categorical features and save encoders
encoders = {}
categorical_cols = df_mergerr.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_merged[col] = le.fit_transform(df_merged[col])
    encoders[col] = le
 
# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
########################################################
 
df_filtered = df_merged.copy()
 
column_to_normalize = 'time_difference_minutes'
df_filtered.loc[:, column_to_normalize] = (df_filtered.loc[:, column_to_normalize] - df_filtered[column_to_normalize].min()) / (df_filtered[column_to_normalize].max() - df_filtered[column_to_normalize].min())
 
df_filtered1 = df_filtered.copy()
 
# Global variables for X and y
global X, y
X = df_filtered1.drop('time_difference_minutes', axis=1)
y = df_filtered1['time_difference_minutes']
 
# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
rf_model = RandomForestRegressor(n_estimators=2, random_state=27, min_samples_split=3)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
 
# User Input and Prediction
g = df_mergerr.drop('time_difference_minutes', axis=1)
 
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
 
# Normalization Functions
def normalize(value, original_min, original_max):
    return (value - original_min) / (original_max - original_min)
 
def reverse_normalize(value):
    return value * (y.max() - y.min()) + y.min()
 
# Get user input

# print(g.columns)
 
# List of string fields
string_fields = ['entity_name', 'modality', 'job_title']
 
# Get string inputs
# for field in string_fields:
#     input_data[field] = input(f"Enter value for {field}: ")
 
# # Get float inputs for other fields
# for column in g.columns:
#     if column not in string_fields:  # Skip string fields
#         while True:
#             try:
#                 value = float(input(f"Enter value for {column}: "))
#                 input_data[column] = value
#                 break
#             except ValueError:
#                 print("Invalid input. Please enter a numeric value.")
 
# # Create a DataFrame from the input data

st.title("Time Prediction model")


entities = ['Select a city'] + df_job_statistics['entity_name'].unique().tolist()
selected_entity = st.selectbox('Select city:', entities)
modality = ['Select a modality'] + df_job_statistics['modality'].unique().tolist()
selected_modality = st.selectbox('Select modality:', modality)
languages = ['Select a language'] + df_job_statistics['job_title'].unique().tolist()
selected_language = st.selectbox('Select language:',languages )

certificate=list(range(3))
job_post_hour=list(range(24))
job_post_minutes=list(range(1000))
job_post_day=list(range(31))

selected_certificate = st.selectbox('Select a certification :', certificate)
selected_jh = st.selectbox('Select a hour :', job_post_hour)
selected_jm = st.selectbox('Select a minute :', job_post_minutes)
selected_jd = st.selectbox('Select a day:', job_post_day)

input_data={
    "entity_name":selected_entity,
    "modality":selected_modality,
    "job_title":selected_language,
    "certificate":selected_certificate,
    "job_post_hour":selected_jh,
    "job_post_minutes":selected_jm,
    "job_post_day":selected_jd,
}

input_df = pd.DataFrame([input_data])


if st.button("Check"):
    # categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])
        else:
            print(f"Warning: Encoder for {col} not found. Skipping this column.")

 
 
# Encode categorical features using saved encoders
 
# Normalize user input
    for column in input_df.columns:
        if column in ["time_difference_minutes"]:
            input_df[column] = normalize(input_df[column], y.min(), y.max())
    
    
    # Make prediction
    prediction = rf_model.predict(input_df)[0]
    prediction_reversed = reverse_normalize(prediction) *1000
    predicted_time_rounded = round((prediction_reversed)/60)
    st.success(f"Predicted time_difference_minutes (reverse normalized and rounded): {predicted_time_rounded} hours")
    st.write(f"R2 score is : {int(r2*100)}%")
 
