"""
Pipeline for 'payamt4_preprocessing'
"""

from kedro.pipeline import Node, Pipeline
from .nodes import preprocess_payamt4


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=preprocess_payamt4,
            inputs="credit_cards_raw",
            outputs="payamt4_preprocessed_data",
            name="preprocess_payamt4_node")
    ])
