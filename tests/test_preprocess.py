import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cust_ml.churn.telco import clean_telco
from cust_ml.preprocess import make_pipeline


def frames():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "c": ["x", "y", "x", "y"]})
    te = pd.DataFrame({"a": [1000.0, 2000.0], "c": ["x", "unseen"]})
    return tr, te, np.array([0, 1, 0, 1])


def test_statistics_come_from_training_data_only():
    tr, te, y = frames()
    pipe = make_pipeline(tr, LogisticRegression()).fit(tr, y)
    num = pipe.named_steps["prep"].named_transformers_["num"]
    assert np.isclose(num.named_steps["scale"].mean_[0], 2.0)  # train mean (median-imputed)
    assert num.named_steps["impute"].statistics_[0] == 2.0  # train median, not test's
    pipe.predict_proba(te)  # transforming test...
    assert np.isclose(num.named_steps["scale"].mean_[0], 2.0)  # ...never refits anything


def test_unseen_category_is_ignored_not_learned():
    tr, te, y = frames()
    pipe = make_pipeline(tr, LogisticRegression()).fit(tr, y)
    cats = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"].categories_[0]
    assert list(cats) == ["x", "y"]
    out = pipe.named_steps["prep"].transform(te)
    assert out[1, -2:].sum() == 0


def test_telco_cleaning_has_no_data_dependent_statistics():
    raw = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "TotalCharges": ["10.5", " "],
            "Churn": ["Yes", "No"],
            "SeniorCitizen": [0, 1],
        }
    )
    X, y = clean_telco(raw)
    assert X["TotalCharges"].isna().tolist() == [False, True]  # blank stays NaN for the pipeline
    assert y.tolist() == [1, 0] and "customerID" not in X
