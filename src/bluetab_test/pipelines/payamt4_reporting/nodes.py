import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import learning_curve

def generate_payamt4_reports(
    df,
    model,
    predictions_df,
    output_dir: str
):
    """
    Genera y guarda gráficas de reporting para el modelo de PAY_AMT4.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Learning Curve
    X = df.drop(columns=["PAY_AMT4", "ID"])
    y = np.log1p(df["PAY_AMT4"])

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = -train_scores.mean(axis=1)
    test_mean = -test_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Train", color="red")
    plt.plot(train_sizes, test_mean, "o-", label="Validation", color="green")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2, color="red")
    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.2, color="green")
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Tamaño del Train")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Predicciones vs Real
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions_df["log_predicted_pay_amt4"], alpha=0.5)
    lims = [0, max(y.max(), predictions_df["log_predicted_pay_amt4"].max())]
    plt.plot(lims, lims, "--", color="red")
    plt.title("Predicción vs Valor Real")
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.grid(True)
    plt.savefig(f"{output_dir}/pred_vs_real.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Histograma de errores
    errors = predictions_df["real_pay_amt4"] - predictions_df["predicted_pay_amt4"]

    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title("Distribución de Errores")
    plt.xlabel("Error (Real - Predicho)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/residual_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_df["predicted_pay_amt4"], errors, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Errores vs Predicción")
    plt.xlabel("Predicción")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig(f"{output_dir}/residual_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Distribución real vs predicha
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions_df["real_pay_amt4"], label="Real", linewidth=2)
    sns.kdeplot(predictions_df["predicted_pay_amt4"], label="Predicho", linewidth=2)
    plt.title("Distribución Real vs Predicha")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Importancia de variables
    try:
        importance = model.feature_importances_
        feature_names = X.columns

        imp_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
              .sort_values("importance", ascending=False)
        )

        plt.figure(figsize=(10, 12))
        sns.barplot(data=imp_df.head(20), y="feature", x="importance")
        plt.title("Top 20 Feature Importances - XGBoost")
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
    except:
        pass

    return f"Reportes generados en: {output_dir}"