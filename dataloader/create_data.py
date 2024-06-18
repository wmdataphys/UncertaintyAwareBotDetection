import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts


def create_dataset(file_path,train_frac=0.7,val_frac=0.5):
    print("Start to load the features")
    feature_df = pd.read_csv(file_path)

    #print(f"Shuffle the features of {len(feature_df)} samples")
    #feature_df = feature_df.sample(frac=1)

    X_df = feature_df.drop(columns = ['class','user_id'])
    x = np.array(X_df)
    y = feature_df['class'].values
    y_binary = (y == 'human').astype(np.float64)
    X_train,X_test_val,y_train,y_test_val = tts(x, y_binary, test_size=0.3, random_state=42)

    X_test,X_val,y_test,y_val = tts(X_test_val, y_test_val, test_size=0.5, random_state=42)

    print("Training fraction: ",len(X_train) / len(x), len(y_train) / len(y))
    print("Validation fraction: ",len(X_val) / len(x), len(y_val) / len(y))
    print("Testing fraction: ",len(X_test) / len(x), len(y_test) / len(y))

    return X_train,X_test,X_val,y_train,y_test,y_val