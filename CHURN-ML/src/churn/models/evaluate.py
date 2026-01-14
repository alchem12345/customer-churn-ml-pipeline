from sklearn.metrics import classification_report,roc_auc_score


# this code will give the accuracy and the classification report of the model 
def evaluate_model(pipeline , X_test , y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]

    metrics  = {
        "roc_auc" : roc_auc_score(y_test,probs),
        "classification_report" : classification_report(y_test,preds)
    }

    return metrics

