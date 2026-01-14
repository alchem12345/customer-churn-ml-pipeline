from sklearn.pipeline import Pipeline

# the actual pipeline for model training 
def pipeline(model , preprocessor):

    return Pipeline(
        steps = [
            ('preprocessing',preprocessor),
        ('model' , model)
        ]
    )
