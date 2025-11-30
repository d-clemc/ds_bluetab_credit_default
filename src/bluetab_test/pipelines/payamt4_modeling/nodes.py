import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import (
    make_scorer, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
from xgboost import XGBRegressor

import matplotlib.pyplot as plt


def train_payamt4_model(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Entrena modelo XGBRegressor para PAY_AMT4.
    Recibe el dataset preprocesado desde el pipeline anterior.
    """

    X = df.drop(columns=["PAY_AMT4", "ID"])
    y = np.log1p(df["PAY_AMT4"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    # model = XGBRegressor(random_state=random_state)
    model = XGBRegressor(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    scoring = {
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'R2': make_scorer(r2_score),
        'RMSE': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
                            greater_is_better=False),
        'Explained Variance': make_scorer(explained_variance_score)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = {
        metric_name: cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
        for metric_name, metric in scoring.items()
    }

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred)
    }

    return model, cv_results, metrics, X_train, X_test, y_train, y_test