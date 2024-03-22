import pandas as pd
from joblib import load

test_samples = pd.read_csv("Labb_ML/test_samples.csv") 
model = load("Labb_ML/classifier_knn_2.pkl") 
#Split
X_test = test_samples.drop("cardio", axis=1)
# Prediction
predictions = model.predict(X_test)

df = pd.DataFrame({"Predictions": predictions})

# Gör förutsägelser med predict_proba
predict_proba = model.predict_proba(X_test)

# Skapa en DataFrame från predict_proba
proba_df = pd.DataFrame(predict_proba, columns=["Probability class 0", "Probability class 1"])

# Lägg till predict_proba i df
df["Probability class 0"] = proba_df["Probability class 0"]
df["Probability class 1"] = proba_df["Probability class 1"]

# Flytta kolumnen "Predictions" till den sista positionen
predictions_column = df.pop("Predictions")
df["Predictions"] = predictions_column
print(df)

