import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st
import altair as alt
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.optimizers import Adam
import pytest

def preprocess_data(data):
	dropped_columns = ['Car Purchase Amount', 'Customer Name', 'Customer e-mail', 'Country']
	X = data.drop(columns=dropped_columns)
	y = data['Car Purchase Amount']
	return X, y, dropped_columns, X.columns.tolist()

data = pd.read_excel('Car_Purchasing_Data.xlsx')

initial_columns = data.columns.tolist()
st.write('Initial columns:', initial_columns)

X, y, dropped_columns, used_columns = preprocess_data(data)

st.write('Dropped columns:', dropped_columns)
st.write('Used columns:', used_columns)

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_reshape = y.values.reshape(-1, 1)
y_scaled = y_scaler.fit_transform(y_reshape)

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=random_state)

model = Sequential([
	Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
	Dropout(0.2),
	Dense(128, activation='relu'),
	Dropout(0.2),
	Dense(1)  #output layer with linear activation for regression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

y_pred = model.predict(X_test)
y_pred_orig = y_scaler.inverse_transform(y_pred).flatten()
y_test_orig = y_scaler.inverse_transform(y_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

st.title('Car Purchase Amount Prediction')
st.write(f'Model RMSE: {rmse}')

df_results = pd.DataFrame({'Actual': y_test_orig, 'Predicted': y_pred_orig})
chart = alt.Chart(df_results).mark_circle().encode(
	x='Actual',
	y='Predicted',
	tooltip=['Actual', 'Predicted']
).properties(
	title='Neural Network Model'
)

st.altair_chart(chart, use_container_width=True)

#test function
def test_preprocess_columns():
	print('aaaaaaaaaaaaaaaaaaaaaaaa')
	columns = ['Customer Name', 'Customer e-mail', 'Country', 'Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']
	data = pd.DataFrame(columns=columns)
	
	_, _, dropped_columns, used_columns = preprocess_data(data)
	
	expected_used_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
	assert used_columns == expected_used_columns, f"Expected used columns {expected_used_columns} but got {used_columns}."
	
	for col in dropped_columns:
		assert col not in used_columns, f"Column {col} should have been dropped but is still present in used columns."
