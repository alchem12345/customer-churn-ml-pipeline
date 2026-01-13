import yaml
from sklearn.linear_model import LogisticRegression

from src.churn.data.load_data import load_raw_data
from src.churn.data.split import split_data
from src.churn.features.preprocessing import build_preprocessor
from src.churn.models.pipeline import build_pipeline
from src.churn.models.train import train_model
from src.churn.models.evaluate import evaluate_model

model_path = "model/churn_pipeline.pkl"

def main():
    config = yaml.safe_load(open("src/churn/config/config.yaml"))

    df = load_raw_data(config["data"]["raw_path"])

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

    pipeline = build_pipeline(preprocessor, model)

    pipeline = train_model(pipeline, X_train, y_train, model)

    metrics = evaluate_model(pipeline, X_test, y_test)
    print(metrics)

if __name__ == "__main__":
    main()