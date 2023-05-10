# Import required modules
from pathlib import Path
from typing import Tuple, Any, Callable
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Set up logging configuration
logger = logging.getLogger("clouds")


def train_model(
    features: pd.DataFrame, model_config: dict
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Trains a model on the provided features using the specified configuration.

    Args:
        features (pd.DataFrame): A Pandas DataFrame containing the features and the target column.
        model_config (dict): A dictionary containing the configuration for the model training process.

    Returns:
        Tuple[Any, pd.DataFrame, pd.DataFrame]: A tuple containing the trained model,
        the train dataset, and the test dataset.
    """

    logger.info("Input dataset shape: %s", features.shape)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        features[model_config["initial_features"]],
        features["class"],
        test_size=model_config["test_train_split"],
    )

    logger.info("Train dataset shape: %s", x_train.shape)
    logger.info("Test dataset shape: %s", x_test.shape)

    # Define a dictionary that maps model names to model classes
    model_mapping = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
    }

    # Get the model_name from the model_config dictionary
    model_name = model_config["model_name"]

    try:
        # Create the model using the mapping
        model_class = model_mapping[model_name]
        model = model_class(**model_config[model_name])
    except KeyError as e:
        logger.error("model does not exist: %s", model_config["model_name"])
        raise KeyError(f"model does not exist: {model_config['model_name']}") from e

    logger.info("Training model with configuration: %s", model_config[model_name])

    # Fit model
    model.fit(x_train, y_train)

    logger.info("Model trained: %s", model_name)

    # Combine data and target labels
    train = x_train.assign(**{"class": y_train})
    test = x_test.assign(**{"class": y_test})

    return model, train, test


def save_file(save_func: Callable, output_file: Path, file_type: str) -> None:
    """
    Saves a file using the specified save function.

    Args:
        save_func (Callable): The function used to save the file.
        output_file (Path): The output file path where the file will be saved.
        file_type (str): The type of file being saved.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If there is a permission issue when trying to save the file.
        pd.errors.EmptyDataError: If the DataFrame is empty and cannot be saved.
        pickle.PicklingError: If there is an error while pickling the model object.

    Returns:
        None
    """
    logger.debug("Attempting to save to %s", output_file)
    try:
        save_func(output_file)
    except FileNotFoundError as e:
        logger.error("The specified directory does not exist: %s", str(e))
    except PermissionError as e:
        logger.error(
            "Permission denied when trying to save the %s to %s: %s",
            file_type,
            output_file,
            str(e),
        )
    except pd.errors.EmptyDataError as e:
        logger.error(
            "The DataFrame is empty and cannot be saved as a %s: %s", file_type, str(e)
        )
    except pickle.PicklingError as e:
        logger.error("An error occurred while pickling the model object: %s", str(e))
    else:
        logger.info("%s saved at %s", file_type.capitalize(), output_file)


def save_data(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    """
    Saves the train and test DataFrames as CSV files in the specified output directory.

    Args:
        train (pd.DataFrame): The train DataFrame to save.
        test (pd.DataFrame): The test DataFrame to save.
        output_dir (Path): The output directory where the files will be saved.

    Returns:
        None
    """
    logger.debug("Attempting to save train test data path: %s", output_dir)

    save_file(train.to_csv, output_dir / "train.csv", "train dataset")
    save_file(test.to_csv, output_dir / "test.csv", "test dataset")


def save_model(trained_model: Any, output_file: Path) -> None:
    """
    Saves a trained model object as a binary file using pickle.

    Args:
        trained_model (Any): The trained model object to save.
        output_file (Path): The output file path where the model will be saved.

    Returns:
        None
    """
    logger.debug("Attempting to save model to path: %s", output_file)
    save_file(
        lambda path: pickle.dump(trained_model, open(path, "wb")),
        output_file,
        "trained model",
    )
