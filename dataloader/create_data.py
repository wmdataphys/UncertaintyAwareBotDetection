import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts


def create_dataset(file_path):
    print("Start to load the features")
    feature_df = pd.read_csv(file_path)

    print(f"Shuffle the features of {len(feature_df)} samples")
    feature_df = feature_df.sample(frac=1)

    X_df = feature_df.drop(columns = ['class','user_id'])
    x = np.array(X_df)

    y = feature_df['class'].values
    y_binary = (y == 'human').astype(np.float64)
    X_train_temp,X_test,y_train_temp,y_test = tts(x, y_binary, test_size=0.2, random_state=42)
    X_train,X_val,y_train,y_val = tts(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

    return X_train,X_test,X_val,y_train,y_test,y_val