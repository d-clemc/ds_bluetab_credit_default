"""
Pipeline for 'payamt4_evaluating'
"""

from kedro.pipeline import Pipeline, Node
from .nodes import evaluate_payamt4_model


def create_pipeline(**kwargs):
    return Pipeline([
        Node(
            func=evaluate_payamt4_model,
            inputs=["payamt4_preprocessed_data", "payamt4_model"],
            outputs=["payamt4_eval_metrics", "payamt4_predictions"],
            name="evaluate_payamt4_model_node",
        )
    ])