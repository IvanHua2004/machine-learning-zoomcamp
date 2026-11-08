"""Train the final model and save it to model.bin.

Usage:
    pipenv run python train.py
"""

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from features import TARGET, add_features, load_raw

DATA_PATH = "data/ai4i2020.csv"
OUTPUT_PATH = "model.bin"
SEED = 42

# TODO(tuning): replace these with the best parameters found in notebook.ipynb
PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "eval_metric": "aucpr",
    "random_state": SEED,
    "n_jobs": -1,
}


def main():
    df = add_features(load_raw(DATA_PATH))

    # Same split as the notebook: 60/20/20, stratified because failures are rare.
    # The final model is trained on train+val and evaluated once on test.
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df[TARGET]
    )

    y_full_train = df_full_train.pop(TARGET).values
    y_test = df_test.pop(TARGET).values

    # TODO(imbalance): decide in the notebook whether to use scale_pos_weight,
    # class weights, or threshold tuning, and apply that choice here.
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        XGBClassifier(**PARAMS),
    )
    pipeline.fit(df_full_train.to_dict(orient="records"), y_full_train)

    y_pred = pipeline.predict_proba(df_test.to_dict(orient="records"))[:, 1]
    print(f"test ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    print(f"test PR AUC:  {average_precision_score(y_test, y_pred):.4f}")

    with open(OUTPUT_PATH, "wb") as f_out:
        pickle.dump(pipeline, f_out)
    print(f"model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
