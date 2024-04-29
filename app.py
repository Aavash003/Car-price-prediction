import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

car_price = pd.read_csv("cleaned_car_datasets.csv",usecols=["model","year","price","transmission","mileage","fuelType","tax","mpg","engineSize","Manufacturer"])
st.write("Dataframe\n")
car_price

data = car_price.copy()

le = LabelEncoder()

data["transmission"] = le.fit_transform(data["transmission"])
data["fuelType"] = le.fit_transform(data["fuelType"])
data["Manufacturer"] = le.fit_transform(data["Manufacturer"])

st.write("\nHeatmap matrix")
correl = data.drop(["model","year"],axis=1).corr()
plt.figure(figsize=(10,7))
sns.heatmap(correl,annot=True,cmap="Reds",cbar=True)
plt.title("Heatmap matrix")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

X = data.drop(["model","year","price"],axis=1)
y = data["price"]


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.1,random_state=100)


rfr = RandomForestRegressor(n_estimators=50,max_depth=15,random_state=100)
rfr.fit(x_train,y_train)
# print("Training score :",rfr.score(x_train,y_train))
# print("Test score",rfr.score(x_test,y_test))

#taking input from user
trans = st.number_input("transmission(0,3)", 0,3)
mileage = st.number_input("mileage(1,323000)", 1, 323000)
fueltype = st.number_input("fueltype(0,4)", 0, 4)
tax = st.number_input("tax(0,580)", 0, 580)
mpg = st.number_input("mpg(.3,470.8)", 0.3, 470.8)
engsize = st.number_input("engsize(0,6.6)", 0.0, 6.6)
manuf = st.number_input("Manufacturer(0,8)", 0, 8)

if st.button("submit"):

	data = {"transmission":[trans],
			"mileage":[mileage],
			"fuelType":[fueltype],
			"tax":[tax],
			"mpg":[mpg],
			"engineSize":[engsize],
			"Manufacturer":[manuf],}


	df = pd.DataFrame(data)
	df
	
	car_price_prediction = rfr.predict(df)
	st.write("Predicted car price : ",car_price_prediction[0])