"""
Pipeline for 'payamt4_modeling'
"""

from kedro.pipeline import Node, Pipeline
from .nodes import train_payamt4_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=train_payamt4_model,
            inputs=["payamt4_preprocessed_data", "params:test_size", "params:random_state"],
            outputs=[
                    "payamt4_model",
                    "payamt4_cv_results",
                    "payamt4_metrics",
                    "X_train_payamt4",
                    "X_test_payamt4",
                    "y_train_payamt4",
                    "y_test_payamt4",
                ],
            name="train_payamt4_model_node"
        )
    ])
