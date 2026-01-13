from sklearn.model_selection import train_test_split

def split_data(df , target_col , test_size , random_state):
    X = df.drop( columns = [target_col])
    y = df[target_col].map({'Yes':1,
                            'No' : 0})
    
    return train_test_split(X,y,test_size=test_size,random_state=random_state)
