import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm._classes import SVC
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- PATHS --- #
DATA_ROOT = Path("data")
TRAIN_DATA_PATH = DATA_ROOT / "train.csv"
TEST_DATA_PATH = DATA_ROOT / "test.csv"

def preprocess(df):
    
    # --- Remove Uneeded Columns ---#
    """
    Justification for removing features:

    Name: Essentially linearly dependent to Sex, a persons name most prominently indicated their sex.
          It could also be an indication of social status, but we already have a feature for that as well.

    PassengerID: This is just an index.

    Ticket: Too much inconsistency in representation amongst samples, also how would you even encode this?

    Cabin: Linearly dependent to Embarked, also how would you even encode this.

    """
    
    # print(df.columns)
    # df = df.dropna()

    y_df = None
    if "Survived" in df.columns:
        y_df = df.loc[:, "Survived"]

    feature_subset = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    
    X_df = df[feature_subset]

    if "PassengerId" in df.columns:
        X_df["PassengerId"] = df.loc[:, "PassengerId"]

    return X_df, y_df
    

def encode(df):
    
    # --- Encode Sex: Male - 0, Female - 1 --- #
    df["Sex"] = df["Sex"].str.strip().str.lower().map({"male": 0, "female": 1})
   

    # --- Encode Embarked: One Hot Encoding --- #
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    
    return df

def train(model, X, y):
    "trains a single model"

    pipe = Pipeline([("scaler", StandardScaler()), model])
    print(pipe)
    breakpoint()

    pipe.fit(X, y)
    return pipe

def get_preds(name, pipe, X_test):
    "evaluates a single model"
    passengerIds = X_test.loc[:, "PassengerId"]
    y_preds = pipe.predict(X_test.drop("PassengerId", axis=1))
    df = pd.concat([pd.DataFrame(passengerIds.values), pd.DataFrame(y_preds)], axis=1)
    df.columns = ["PassengerId", "Survived"]
    return df


def main():    

    # --- Load Data --- #
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    train_X_df, train_y_df = preprocess(train_df)

    train_X_df = encode(train_X_df)

    # --- Load Testing --- #
    test_df = pd.read_csv(TEST_DATA_PATH)

    test_X_df, _ = preprocess(test_df)

    test_X_df = encode(test_X_df)

    test_X_df.loc[152, "Fare"] = test_X_df["Fare"].mean(numeric_only=True)

    # --- Train --- #

    models = {
        "Logistic": LogisticRegression(),
        "SVM": SVC(),
        "DecisionTree": DecisionTreeClassifier(),
    }

    for name, model in models.items():
        pipe = train(model, train_X_df, train_y_df)
        predictions = get_preds(name, pipe, test_X_df)
        predictions.to_csv(f"{name}_preds.csv", index=False)



if __name__ == "__main__":
    main()
