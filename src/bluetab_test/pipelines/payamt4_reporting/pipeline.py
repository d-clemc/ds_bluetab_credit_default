"""
Pipeline for 'payamt4_reporting'
"""

from kedro.pipeline import Pipeline, Node
from .nodes import generate_payamt4_reports

def create_pipeline(**kwargs):
    return Pipeline([
        Node(
            func=generate_payamt4_reports,
            inputs=[
                "payamt4_preprocessed_data",
                "payamt4_model",
                "payamt4_predictions",
                "params:report_output_dir"
            ],
            outputs="payamt4_reporting",
            name="generate_payamt4_reports_node"
        )
    ])