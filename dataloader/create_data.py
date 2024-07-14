import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

# def create_dataset(file_path,train_frac=0.7,val_frac=0.5):
#     print("Start to load the features")
#     feature_df = pd.read_csv(file_path,sep=',',index_col=None)


#     X_df = feature_df.drop(columns = ['class','user_id'])
#     x = np.array(X_df)
#     y = feature_df['class'].values
#     y_binary = (y == 'bot').astype(np.float64)
#     X_train,X_test_val,y_train,y_test_val = tts(x, y_binary, test_size=1.0 - train_frac, random_state=42)
#     X_test,X_val,y_test,y_val = tts(X_test_val, y_test_val, test_size=val_frac, random_state=42)

#     print("Training fraction: ",len(X_train) / len(x), len(y_train) / len(y))
#     print("Validation fraction: ",len(X_val) / len(x), len(y_val) / len(y))
#     print("Testing fraction: ",len(X_test) / len(x), len(y_test) / len(y))

#     return X_train,X_test,X_val,y_train,y_test,y_val

def create_dataset(file_path, train_frac=0.7, val_frac=0.5,random_seed=8,leftover_accounts=False,method=None):
    if method is None:
        print("Method is none, please specify.")
        exit()

    print("Start to load the features")
    feature_df = pd.read_csv(file_path)
    if method == "BLOC":
        X_df = feature_df.drop(columns=['class', 'userID'])
    else:
        X_df = feature_df.drop(columns=['class','label','userID','src'])

    x = np.array(X_df)
    y = feature_df['class'].values
    y_binary = (y == 'bot').astype(np.float64)

    # Separate bot and human data
    bot_indices = np.where(y_binary == 1.0)[0]
    human_indices = np.where(y_binary == 0.0)[0]

    if method == "BOTOMETER":
        print("Using Z-Score scaler for " + str(method))
        scaler = StandardScaler()
        scaler = scaler.fit(x)
        x = scaler.transform(x) # scaling off globals, ok since low stats

    # Determine the number of samples to match the minority class
    n_humans = len(human_indices)
    n_bots = len(bot_indices)
    np.random.seed(random_seed) # Set random seed again just incase global does not hold. It should.
    if n_bots > n_humans:
        print("Undersampling bots.")
        bot_sample_indices = np.random.choice(bot_indices, n_humans, replace=False)
        human_sample_indices = human_indices
    else:
        print("Undersampling humans.")
        bot_sample_indices = bot_indices
        human_sample_indices = np.random.choice(human_indices, n_bots, replace=False)

    balanced_indices = np.concatenate([bot_sample_indices, human_sample_indices])
    np.random.shuffle(balanced_indices)

    x_balanced = x[balanced_indices]
    y_balanced = y_binary[balanced_indices]

    X_train, X_test_val, y_train, y_test_val = tts(x_balanced, y_balanced, test_size=1.0 - train_frac, random_state=42)
    X_test, X_val, y_test, y_val = tts(X_test_val, y_test_val, test_size=val_frac, random_state=42)

    if n_bots > n_humans:
        removed_account_indices = np.setdiff1d(bot_indices, bot_sample_indices)
        X_removed_accounts = x[removed_account_indices]
        y_removed_accounts = y_binary[removed_account_indices]
        account_type = 'bot'

    else:
        removed_account_indices = np.setdiff1d(human_indices, human_sample_indices)
        X_removed_accounts = x[removed_account_indices]
        y_removed_accounts = y_binary[removed_account_indices] 
        account_type = 'human'

    print("Training fraction: ", len(X_train) / len(x), len(y_train) / len(y))
    print("Validation fraction: ", len(X_val) / len(x), len(y_val) / len(y))
    print("Testing fraction: ", len(X_test) / len(x), len(y_test) / len(y))

    if leftover_accounts:
        return X_train, X_test, X_val, y_train, y_test, y_val, X_removed_accounts, y_removed_accounts, account_type
    else:
        return X_train,X_test,X_val,y_train,y_test,y_val
