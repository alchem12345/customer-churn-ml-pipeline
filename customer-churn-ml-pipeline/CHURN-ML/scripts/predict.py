import pandas as pd
from src.churn.inference.predict import load_model, predict

MODEL_PATH = "models/churn_pipeline.pkl"

def main():
    model = load_model(MODEL_PATH)

    sample = pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.3,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }])

    prediction = predict(model, sample)
    print("Churn prediction:", prediction)

if __name__ == "__main__":
    main()
