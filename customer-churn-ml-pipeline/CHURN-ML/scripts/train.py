import yaml
from sklearn.linear_model import LogisticRegression

from churn.data.load_data import load_data
from churn.data.split import split_data
from churn.features.preprocessing import build_preprocessor
from churn.models.pipeline import pipeline
from churn.models.train import train
from churn.models.evaluate import evaluate_model

model_path = "model/churn_pipeline.pkl"

def main():
    config = yaml.safe_load(open("src/churn/config/config.yaml"))

    df = load_data(config["data"]["raw_path"])

    X_train, X_test, y_train, y_test = split_data(
        df,
        config["data"]["target_col"],
        config["data"]["test_size"],
        config["data"]["random_state"],
    )

    preprocessor = build_preprocessor(
        config["features"]["numeric"],
        config["features"]["categorical"]
    )

    model = LogisticRegression(**config["model"]["params"])

    churn_pipeline = pipeline(preprocessor, model)

    churn_pipeline = train(churn_pipeline, X_train, y_train, model_path)

    metrics = evaluate_model(churn_pipeline, X_test, y_test)
    print(metrics)

if __name__ == "__main__":
    main()
