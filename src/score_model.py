# Import required modules
from pathlib import Path
from typing import Tuple, Any
import logging
import pandas as pd
import numpy as np

# Set up logging configuration
logger = logging.getLogger("clouds")


def score_model(
    test: pd.DataFrame, tmo: Any, score_config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scores the test dataset using the trained model object.

    Args:
        test (pd.DataFrame): A Pandas DataFrame of the test dataset.
        tmo (Any): The trained model object.
        score_config (dict): A dictionary containing the configuration for scoring.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the predicted probabilities and
        predicted binary labels for the test dataset.
    """
    logger.info("Started scoring test data")
    try:
        # Get the probability threshold from the score_config dictionary
        threshold = score_config.get("probability_threshold", 0.5)

        # Use the trained model object tmo to predict the probabilities of the test set
        ypred_proba_test = tmo.predict_proba(test.drop(columns="class", axis=1))[:, 1]

        # Convert the probabilities to binary labels based on the threshold
        ypred_bin_test = (ypred_proba_test >= threshold).astype(int)

        # Log variable details
        logger.debug("Test dataset shape: %s", test.shape)
        logger.debug("Probability threashhold: %s", threshold)
        logger.debug("Dropped %d columns for scoring", len(test.columns) - 1)
        logger.debug("Number of true labels: %s", sum(ypred_bin_test))

        logger.info("Test data scored successfully")
        return test["class"].to_numpy(), ypred_proba_test, ypred_bin_test

    except AttributeError as e:
        logger.error("Error occurred while scoring the test data: %s", e)
        raise AttributeError("Error occurred while scoring the test data") from e
    except KeyError as e:
        logger.error("Error occurred while accessing class column: %s", e)
        raise KeyError("Error occurred while accessing class column") from e


def save_scores(
    scores: Tuple[np.ndarray, np.ndarray, np.ndarray], output_file: Path
) -> None:
    """
    Saves the true class labels, predicted probabilities, and binary labels to a CSV file.

    Args:
        scores (Tuple[np.ndarray, np.ndarray, np.ndarray]): A tuple containing the true class labels,
        predicted probabilities, and predicted binary labels.
        output_file (Path): The output file where the scores will be saved.
    """
    logger.info("Saving scores to output file %s", output_file)

    # Unpack the scores tuple
    true_labels, ypred_proba_test, ypred_bin_test = scores

    logger.debug(
        "Scores tuple shape: %s",
        (len(true_labels), len(ypred_proba_test), len(ypred_bin_test)),
    )

    # Create a pandas DataFrame to hold the scores
    scores_df = pd.DataFrame(
        {
            "true_label": true_labels,
            "probability": ypred_proba_test,
            "prediction": ypred_bin_test,
        }
    )

    logger.debug("First few rows of the scores DataFrame:\n%s", scores_df.head())
    logger.debug("Scores DataFrame shape: %s", scores_df.shape)

    logger.debug("Attempting to save scores to path: %s", output_file)

    try:
        # Save the DataFrame to a CSV file
        scores_df.to_csv(output_file, index=False)
    except FileNotFoundError as e:
        logger.error("The specified directory does not exist: %s", str(e))
    except PermissionError as e:
        logger.error(
            "Permission denied when trying to save the scores to %s: %s",
            output_file,
            str(e),
        )
    except pd.errors.EmptyDataError as e:
        logger.error(
            "The DataFrame is empty and cannot be saved as a CSV file: %s", str(e)
        )
    else:
        logger.info("Scores saved to output file: %s", output_file)
