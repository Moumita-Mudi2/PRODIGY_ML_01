import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Moumita Mudi\OneDrive\Desktop\jupyter projects\house-prices\train.csv")

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
X = data[features]
y = data["SalePrice"]

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

st.set_page_config(page_title="House Price Predictor",page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("This app predicts house prices using a Linear Regression model.")

if st.checkbox("Show Dataset"):
    st.write("### Dataset")
    st.dataframe(data.head())

st.write("### Model Performance Metrics")
st.write(f"- Mean Squared Error: **{mse:.2f}**")
st.write(f"- R-Squared Value: **{r2:.2f}**")

st.write("### Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test , y_pred , alpha=0.5)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual Prices vs Predicted Prices")
st.pyplot(fig)

st.write("### Predict House Price")
GrLivArea=st.number_input("Enter Ground Living Area(in sq ft)",min_value=0,value=2000)
BedroomAbvGr = st.number_input("Enter Number of Bedrooms",min_value=0,value=3)
FullBath=st.number_input("Enter Number of Bathrooms",min_value=0,value=2)

if st.button("Predict Price"):
    new_data = pd.DataFrame({'GrLivArea':[GrLivArea],'BedroomAbvGr':[BedroomAbvGr],'FullBath':[FullBath] })
    predicted_price = model.predict(new_data)
    st.success(f"Predicted Price:**${predicted_price[0]:,.2f}**")