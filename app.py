
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

st.title("🍷 Wine Quality Predictor (KNN)")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    return pd.read_csv(url, sep=';')

data = load_data()
st.write("### Dataset Preview", data.head())

X = data.drop("quality", axis=1)
y = data["quality"]

normalize = st.checkbox("Normalize features", value=True)

if normalize:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

k = st.slider("Select K (number of neighbors)", 1, 15, 5)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {acc:.2f}")

st.write("## Enter Wine Characteristics to Predict Quality")
input_data = {}

for col in X.columns:
    val = st.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    input_data[col] = val

if st.button("Predict Wine Quality"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df) if normalize else input_df.values
    prediction = knn.predict(input_scaled)
    st.success(f"Predicted Wine Quality: **{prediction[0]}**")
