# Import required modules
import logging
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score,
)

# Set up logging configuration
logger = logging.getLogger("clouds")


def evaluate_performance(
    scores: Tuple[np.ndarray, np.ndarray, np.ndarray], evaluate_config: dict
) -> Dict[str, float]:
    """Evaluates the performance of a binary classification model on a test set.

    Args:
        scores: A tuple containing the true binary labels, predicted probabilities,
        and predicted binary labels for the test set.

        evaluate_config: A dictionary specifying the evaluation options.
        The dictionary should contain the following key: "metrics".
        The value of this key should be a list of strings indicating which performance
        metrics to evaluate. Additionally, it may contain the following key: "log_confusion_matrix".
        The value of this key should be a boolean indicating whether to log the confusion matrix.

    Returns:
        A dictionary containing the performance metrics of the model on the test set.
        The dictionary contains the following keys:
        "AUC", "confusion_matrix", "accuracy", "f1_score", and "classification_report".
        The values associated with each key are the corresponding performance metrics.

    Raises:
        ValueError: If the "metrics" key in evaluate_config is not a list of strings.
    """

    # Unpack scores into variables
    y_test, ypred_proba_test, ypred_bin_test = scores
    logger.debug("Unpacked scores")

    # Define metric functions
    metric_functions = {
        "AUC": calc_auc,
        "confusion_matrix": calc_confusion_matrix,
        "accuracy": calc_accuracy,
        "f1_score": calc_f1_score,
        "classification_report": calc_classification_report,
    }
    logger.debug("Defined metric functions")

    # Get list of metrics to evaluate and init dictionary
    metrics_to_run = evaluate_config.get("metrics", metric_functions.keys())
    metrics = {}
    logger.debug("Initialized metrics dictionary")

    # Define arrays for each metric
    metric_args = {
        "AUC": (y_test, ypred_proba_test),
        "confusion_matrix": (y_test, ypred_bin_test),
        "accuracy": (y_test, ypred_bin_test),
        "f1_score": (y_test, ypred_bin_test),
        "classification_report": (y_test, ypred_bin_test),
    }

    # Evaluate each metric
    for metric in metrics_to_run:
        logger.debug("Evaluating metric: %s", metric)
        if metric in metric_functions:
            try:
                # Call metric function and store result
                metrics[metric] = metric_functions[metric](*metric_args[metric])
                logger.debug("Calculated %s: %s", metric, metrics[metric])

                # Log result
                if metric == "confusion_matrix":
                    logger.info("%s on test:\n %s", metric, np.array(metrics[metric]))
                elif metric == "classification_report":
                    logger.info(
                        "%s on test:\n %s",
                        metric,
                        pd.DataFrame(metrics[metric]).transpose(),
                    )
                else:
                    logger.info("%s on test: %0.3f", metric, metrics[metric])
            except ValueError as e:
                # Log error message if metric function raises ValueError
                logger.error("Error occurred during %s calculation: %s", metric, e)
        else:
            # Log warning message if metric is not recognized
            logger.warning("Invalid metric specified: %s", metric)

    logger.debug("Finished evaluation of model performance")
    return metrics


# Helper functions


def calc_auc(y_test, ypred_proba_test):
    """Calculates the area under the receiver operating characteristic (ROC) curve."""
    return roc_auc_score(y_test, ypred_proba_test)


def calc_confusion_matrix(y_test, ypred_bin_test):
    """Calculates the confusion matrix."""
    return confusion_matrix(y_test, ypred_bin_test).tolist()


def calc_accuracy(y_test, ypred_bin_test):
    """Calculates the accuracy."""
    return accuracy_score(y_test, ypred_bin_test)


def calc_f1_score(y_test, ypred_bin_test):
    """Calculates the F1 score."""
    return f1_score(y_test, ypred_bin_test)


def calc_classification_report(y_test, ypred_bin_test):
    """Calculates the calc_classification_report"""
    return classification_report(y_test, ypred_bin_test, output_dict=True)


def save_metrics(metrics, directory: Path) -> None:
    """
    Saves the evaluation metrics to a YAML file in the specified directory.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics to save.
        directory (Path): A Path object representing the directory where the YAML file will be saved.

    Returns:
        None

    Raises:
        -
    """
    logger.debug("Converting numbers for saving metrics")
    # Convert numpy scalar values to native Python data types
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            metrics[key] = value.item()

    logger.debug("Saving metrics")
    try:
        # Save the metrics as a YAML file
        with open(directory, "w") as f:
            yaml.dump(
                metrics,
                f,
                default_flow_style=False,  # Maintain readability with block-style format
                sort_keys=False,  # Keep the original order of keys
                allow_unicode=True,  # Allow special characters
            )
    except FileNotFoundError as e:
        logger.error("The specified directory does not exist: %s", str(e))
    except PermissionError as e:
        logger.error(
            "Permission denied when trying to save the metrics to %s: %s",
            directory,
            str(e),
        )
    except yaml.YAMLError as e:
        logger.error(
            "Error occurred while serializing the metrics to YAML format: %s", str(e)
        )
