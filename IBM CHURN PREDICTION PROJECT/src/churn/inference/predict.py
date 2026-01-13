import joblib

def load_model(path):
    return joblib.load(path)

def predict(model , input):
    return model.predict(input)
    
