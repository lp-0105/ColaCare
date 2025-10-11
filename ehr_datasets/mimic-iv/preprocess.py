import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ehr_datasets.utils.tools import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time, export_note

processed_data_dir = os.path.join("./ehr_datasets/mimic-iv", 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Record feature names
basic_records = ['RecordID', 'PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age']
labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# Stratified split dataset into train, validation and test sets
# For ml/dl models: include Imputation & Normalization & Outlier Filtering steps
# For all settings, randomly select 200 patients for test set
# Then randomly select 10000 in the rest used for training and validation (7/8 training, 1/8 validation)

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic4_formatted_ehr.parquet'))
if "RecordID" not in df.columns:
    df["RecordID"] = df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str)
df = df[basic_records + target_features + demographic_features + labtest_features]

# Ensure the data is sorted by RecordID and RecordTime
df = df.sort_values(by=['RecordID', 'RecordTime']).reset_index(drop=True)

# Group the dataframe by `RecordID`
grouped = df.groupby('RecordID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=1/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=4/99, random_state=SEED, stratify=train_val_patients_outcome)
test_df = df[df['RecordID'].isin(test_patients)]

# For llm setting, export data on test set:
# Export the missing mask
train_missing_mask = export_missing_mask(df, demographic_features, labtest_features, id_column='RecordID')
val_missing_mask = export_missing_mask(df, demographic_features, labtest_features, id_column='RecordID')
test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features, id_column='RecordID')

# Export the record time
train_record_time = export_record_time(df, id_column='RecordID')
val_record_time = export_record_time(df, id_column='RecordID')
test_record_time = export_record_time(test_df, id_column='RecordID')

# Export the raw data
_, train_raw_x, _, _ = forward_fill_pipeline(df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')
_, val_raw_x, _, _ = forward_fill_pipeline(df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')

# Create the directory to save the processed data
save_dir = os.path.join(processed_data_dir, 'processed', 'fold_1')
os.makedirs(save_dir, exist_ok=True)

pd.to_pickle(train_missing_mask, os.path.join(save_dir, 'train_missing_mask.pkl'))
pd.to_pickle(val_missing_mask, os.path.join(save_dir, 'val_missing_mask.pkl'))
pd.to_pickle(test_missing_mask, os.path.join(save_dir, 'test_missing_mask.pkl'))
pd.to_pickle(train_record_time, os.path.join(save_dir, 'train_record_time.pkl'))
pd.to_pickle(val_record_time, os.path.join(save_dir, 'val_record_time.pkl'))
pd.to_pickle(test_record_time, os.path.join(save_dir, 'test_record_time.pkl'))
pd.to_pickle(train_raw_x, os.path.join(save_dir, 'train_raw_x.pkl'))
pd.to_pickle(val_raw_x, os.path.join(save_dir, 'val_raw_x.pkl'))
pd.to_pickle(test_raw_x, os.path.join(save_dir, 'test_raw_x.pkl'))

# For ml/dl models, convert categorical features to one-hot encoding
one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
columns = df.columns.to_list()
column_start_idx = columns.index(categorical_labtest_features[0])
column_end_idx = columns.index(categorical_labtest_features[-1])
df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)

# Update the categorical lab test features
ehr_categorical_labtest_features = one_hot.columns.to_list()
ehr_labtest_features = ehr_categorical_labtest_features + numerical_labtest_features
require_impute_features = ehr_labtest_features

# Group the dataframe by patient ID
grouped = df.groupby('RecordID')

# Print the sizes of the datasets
print("Train patients size:", len(train_patients))
print("Validation patients size:", len(val_patients))
print("Test patients size:", len(test_patients))

# Assert there is no data leakage
assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage between train and val sets"
assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage between train and test sets"
assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage between val and test sets"

# Create train, val dataframes
train_df = df[df['RecordID'].isin(train_patients)]
val_df = df[df['RecordID'].isin(val_patients)]
test_df = df[df['RecordID'].isin(test_patients)]

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features, id_column="RecordID")

# Forward Imputation after grouped by RecordID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)
pd.to_pickle(['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation','Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH'], os.path.join(save_dir, 'labtest_features.pkl'))

pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))
pd.to_pickle(df[['PatientID', 'Sex', 'Age']].groupby('PatientID').first().to_dict('index'), os.path.join(save_dir, 'basic.pkl'))