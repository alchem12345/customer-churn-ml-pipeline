import joblib

def train(pipeline ,X_train , Y_train , model_path):
    pipeline.fit(X_train,Y_train)
    joblib.dump(pipeline,model_path)

    return pipeline

    