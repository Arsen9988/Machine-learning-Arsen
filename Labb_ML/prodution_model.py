import pandas as pd
from joblib import load

test_samples = pd.read_csv("Labb_ML/test_samples.csv") 
model = load("Labb_ML/classifier_knn_2.pkl") 
X_test = test_samples.drop("cardio", axis=1)
predictions = model.predict(X_test)
# Ny df
df = pd.DataFrame({"Predictions": predictions})
# Gör förutsägelser med predict_proba
predict_proba = model.predict_proba(X_test)
# Ny df från predict_proba
proba_df = pd.DataFrame(predict_proba, columns=["Probability class 0", "Probability class 1"])
# Lägger till de nya kolumnerna, predict_proba i df
df["Probability class 0"] = proba_df["Probability class 0"]
df["Probability class 1"] = proba_df["Probability class 1"]
# Flyttar kolumnen "Predictions" till den sista positionen
predictions_column = df.pop("Predictions")
df["Predictions"] = predictions_column
# Sparar prediktioner med sannolikheter
df.to_csv("Labb_ML/prediction.csv", index=False)