import pandas as pd
from scipy.stats.mstats import winsorize


def preprocess_payamt4(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocesa el dataset para el modelo de regresión de PAY_AMT4,
    evitando data leakage y generando features derivadas.

    Pasos:
    - Selección de columnas permitidas (solo info hasta mayo)
    - Winsorizing en montos (1%-99%)
    - Feature engineering (bill/pago, tendencias, retrasos parciales)
    - Limpieza de categorías EDUCATION y MARRIAGE
    - Manejo de NaNs en ratios

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo de UCI (default of credit card clients).
    Returns
    -------
    pd.DataFrame
        DataFrame preprocesado listo para modelado.
    """

    allowed_cols = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_6",
        "PAY_5",
        "BILL_AMT6",
        "BILL_AMT5",
        "PAY_AMT6",
        "PAY_AMT5",
    ]

    df_no_leakage = df[allowed_cols + ["PAY_AMT4"]].copy()

    # Winsorizing (1%) en montos
    winsor_cols = ["BILL_AMT6", "BILL_AMT5", "PAY_AMT6", "PAY_AMT5"]

    for col in winsor_cols:
        df_no_leakage[col] = winsorize(df_no_leakage[col], limits=[0.01, 0.01])

    # Feature Engineering (solo abril y mayo)

    # Total y promedio de bill
    df_no_leakage["total_bill"] = df_no_leakage[["BILL_AMT6", "BILL_AMT5"]].sum(axis=1)
    df_no_leakage["avg_bill"] = df_no_leakage[["BILL_AMT6", "BILL_AMT5"]].mean(axis=1)

    # Tendencia de bill (mayo - abril)
    df_no_leakage["bill_trend"] = df_no_leakage["BILL_AMT5"] - df_no_leakage["BILL_AMT6"]

    # Ratios de pago por mes
    df_no_leakage["pay_ratio_6"] = df_no_leakage["PAY_AMT6"] / (df_no_leakage["BILL_AMT6"] + 1)
    df_no_leakage["pay_ratio_5"] = df_no_leakage["PAY_AMT5"] / (df_no_leakage["BILL_AMT5"] + 1)

    pay_ratio_cols = ["pay_ratio_6", "pay_ratio_5"]
    df_no_leakage["avg_pay_ratio"] = df_no_leakage[pay_ratio_cols].mean(axis=1)
    df_no_leakage["min_pay_ratio"] = df_no_leakage[pay_ratio_cols].min(axis=1)
    df_no_leakage["max_pay_ratio"] = df_no_leakage[pay_ratio_cols].max(axis=1)

    # Tendencia de pago (mayo - abril)
    df_no_leakage["payment_trend"] = df_no_leakage["PAY_AMT5"] - df_no_leakage["PAY_AMT6"]

    # Retrasos parciales (solo PAY_6 y PAY_5)
    pay_status_cols = ["PAY_6", "PAY_5"]
    df_no_leakage["delay_sum_partial"] = df_no_leakage[pay_status_cols].clip(lower=0).sum(axis=1)
    df_no_leakage["delay_count_partial"] = (df_no_leakage[pay_status_cols] > 0).sum(axis=1)
    df_no_leakage["max_delay_partial"] = df_no_leakage[pay_status_cols].max(axis=1)

    # 4. Limpieza categórica
    df_no_leakage["EDUCATION"] = df_no_leakage["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df_no_leakage["MARRIAGE"] = df_no_leakage["MARRIAGE"].replace({0: 3})

    # Manejo de NaNs (principalmente en ratios)
    df_no_leakage = df_no_leakage.fillna(0)

    return df_no_leakage