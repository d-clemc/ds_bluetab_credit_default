import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

def evaluate_payamt4_model(df: pd.DataFrame, model) -> tuple:
    """
    Evalúa el modelo de PAY_AMT4 ya entrenado.

    - Usa el modelo guardado (.pkl)
    - Predice PAY_AMT4 en escala logarítmica
    - Aplica expm1() para llevarlo a escala original
    - Calcula métricas (MSE, MAE, RMSE, R2, ExplainedVariance)
    """

    X = df.drop(columns=["PAY_AMT4", "ID"])
    y_true = df["PAY_AMT4"] 

    y_pred_log = model.predict(X)

    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Explained Variance": ev,
    }

    predictions_df = pd.DataFrame({
        "ID": df["ID"],
        "real_pay_amt4": y_true,
        "predicted_pay_amt4": y_pred,
        "log_predicted_pay_amt4": y_pred_log
    })

    return metrics, predictions_df