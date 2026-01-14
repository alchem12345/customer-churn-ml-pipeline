from sklearn.pipeline import Pipeline

# the actual pipeline for model training
def pipeline(preprocessor, model):

    return Pipeline(
        steps = [
            ('preprocessing',preprocessor),
        ('model' , model)
        ]
    )
