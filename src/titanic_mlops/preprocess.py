import pandas as pd
from pathlib import Path

# File paths
RAW_DATA = Path("data/raw/train.csv")
PROCESSED_DATA = Path("data/processed/train.csv")


def main():
    # Load dataset
    df = pd.read_csv(RAW_DATA)

    # Drop columns that won't be used
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Save processed data
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()