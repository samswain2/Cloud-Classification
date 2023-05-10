# Import required modules
from typing import Dict
import logging
import numpy as np
import pandas as pd

# Set up logging configuration
logger = logging.getLogger("clouds")


class FeatureTransformationError(Exception):
    """Custom exception class for feature transformation-related errors."""


def generate_features(features: pd.DataFrame, features_config: Dict) -> pd.DataFrame:
    """
    Applies a set of feature transformations to a given set of features, based on the configuration specified in
    `features_config`.

    Args:
        features (pd.DataFrame): The input features to transform.
        features_config (Dict): A dictionary specifying the feature transformations to apply. The dictionary should
                                contain one or more of the following keys: "calculate_norm_range", "log_transform",
                                and "multiply". The value of each key should be a dictionary containing the parameters
                                required for that transformation.

    Returns:
        pd.DataFrame: The transformed features.
    """

    logger.info("Starting feature generation")

    transformation_mapping = {
        "calculate_norm_range": calculate_norm_range,
        "log_transform": log_transform,
        "multiply": multiply_columns,
        "standard_scale": standard_scale,
        "sqrt_transform": sqrt_transform,
    }

    for transformation_key, transformation_params in features_config.items():
        transformation_function = transformation_mapping[transformation_key]
        features = transformation_function(features, transformation_params)

    logger.info("Feature generation complete")

    return features


# Transformation functions
def calculate_norm_range(features: pd.DataFrame, norm_feats: Dict) -> pd.DataFrame:
    """
    Calculates the normalized range for the specified columns in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        norm_feats (Dict): A dictionary specifying the columns for which to calculate the normalized range.
                           The dictionary keys are the names of the new columns and the values are dictionaries
                           containing keys for the min, max, and mean columns.

    Returns:
        pd.DataFrame: The DataFrame with the new normalized range columns added.
    """

    logger.info("Starting 'calculate_norm_range' transformation")

    for col_range in norm_feats:
        logger.info("Processing column range '%s'", col_range)

        try:
            min_col = norm_feats[col_range]["min_col"]
            max_col = norm_feats[col_range]["max_col"]
            mean_col = norm_feats[col_range]["mean_col"]
            features[col_range] = calculate_norm_range_column(
                features, min_col, max_col, mean_col
            )
        except KeyError as e:
            logger.error(
                "Error during 'calculate_norm_range' transformation for '%s': %s",
                col_range,
                str(e),
            )
            continue
        logger.info("Successfully processed column range '%s'", col_range)

    logger.info("'calculate_norm_range' transformation complete")

    return features


def log_transform(features: pd.DataFrame, log_feats: Dict) -> pd.DataFrame:
    """
    Applies log transformation to the specified columns in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        log_feats (Dict): A dictionary specifying the columns to apply the log transformation.
                          The dictionary keys are the names of the new columns and the values are either
                          the original column names or dictionaries containing the original column name
                          and an optional epsilon value.

    Returns:
        pd.DataFrame: The DataFrame with the log-transformed columns added.
    """

    logger.info("Starting 'log_transform' transformation")

    for log_col, col_config in log_feats.items():
        logger.info("Processing column '%s'", log_col)

        try:
            if isinstance(col_config, dict):
                epsilon = col_config.get("epsilon", 1e-10)
            else:
                epsilon = 1e-10
            features[log_col] = log_transform_column(features[col_config], epsilon)
        except KeyError as e:
            logger.error(
                "Error during 'log_transform' transformation for '%s': %s",
                log_col,
                str(e),
            )
            continue
        logger.info("Successfully processed column '%s'", log_col)

    logger.info("'log_transform' transformation complete")

    return features


def multiply_columns(features: pd.DataFrame, mult_features: Dict) -> pd.DataFrame:
    """
    Multiplies pairs of columns element-wise in the input DataFrame and creates new columns for the results.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        mult_features (Dict): A dictionary specifying the pairs of columns to multiply.
                              The dictionary keys are the names of the new columns and the values are dictionaries
                              containing the keys "col_a" and "col_b" for the columns to multiply.

    Returns:
        pd.DataFrame: The DataFrame with the new multiplied columns added.
    """
    logger.info("Starting 'multiply_columns' transformation")

    for mult_cols in mult_features:
        logger.info("Processing columns '%s'", mult_cols)

        try:
            col_a = mult_features[mult_cols]["col_a"]
            col_b = mult_features[mult_cols]["col_b"]
            features[mult_cols] = multiply_columns_elementwise(features, col_a, col_b)
        except KeyError as e:
            logger.error(
                "Error during 'multiply_columns' transformation for '%s': %s",
                mult_cols,
                str(e),
            )
            continue
        logger.info("Successfully processed columns '%s'", mult_cols)

    logger.info("'multiply_columns' transformation complete")

    return features


def standard_scale(features: pd.DataFrame, scale_feats: Dict) -> pd.DataFrame:
    """
    Applies standard scaling to the specified columns in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        scale_feats (Dict): A dictionary specifying the columns to apply the standard scaling.

    Returns:
        pd.DataFrame: The DataFrame with the standardized columns added.
    """

    logger.info("Starting 'standard_scale' transformation")

    for col in scale_feats:
        logger.info("Processing column '%s'", col)
        try:
            features[col] = standard_scale_column(features[col])
        except KeyError as e:
            logger.error(
                "Error during 'standard_scale' transformation for '%s': %s", col, str(e)
            )
            continue
        logger.info("Successfully processed column '%s'", col)

    logger.info("'standard_scale' transformation complete")
    return features


def sqrt_transform(features: pd.DataFrame, sqrt_feats: Dict) -> pd.DataFrame:
    """
    Applies square root transformation to the specified columns in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        sqrt_feats (Dict): A dictionary specifying the columns to apply the square root transformation.
                           The dictionary keys are the names of the new columns and the values are
                           the original column names.

    Returns:
        pd.DataFrame: The DataFrame with the square root-transformed columns added.
    """

    logger.info("Starting 'sqrt_transform' transformation")

    for sqrt_col, orig_col in sqrt_feats.items():
        logger.info("Processing column '%s'", sqrt_col)
        try:
            features[sqrt_col] = sqrt_transform_column(features[orig_col])
        except KeyError as e:
            logger.error(
                "Error during 'sqrt_transform' transformation for '%s': %s",
                orig_col,
                str(e),
            )
            continue
        logger.info("Successfully processed column '%s'", sqrt_col)

    logger.info("'sqrt_transform' transformation complete")

    return features


# Helper functions
def calculate_norm_range_column(
    features: pd.DataFrame, min_col: str, max_col: str, mean_col: str
) -> pd.DataFrame:
    """
    Calculates the normalized range for a single column in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        min_col (str): The column name for the minimum values.
        max_col (str): The column name for the maximum values.
        mean_col (str): The column name for the mean values.

    Returns:
        pd.DataFrame: A DataFrame containing the normalized range for the specified column.
    """
    range_col = features[max_col] - features[min_col]
    return range_col.divide(features[mean_col])


def log_transform_column(col: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Applies log transformation to a single column in the input DataFrame.

    Args:
        col (pd.DataFrame): The input column to apply the log transformation.
        epsilon (float): A small constant to add before taking the log to avoid taking the log of 0.

    Returns:
        pd.DataFrame: A DataFrame containing the log-transformed values of the input column.
    """
    return col.apply(lambda x: np.log1p(x + epsilon))


def multiply_columns_elementwise(
    features: pd.DataFrame, col_a: str, col_b: str
) -> pd.DataFrame:
    """
    Multiplies two columns element-wise in the input DataFrame.

    Args:
        features (pd.DataFrame): The input features DataFrame.
        col_a (str): The name of the first column to multiply.
        col_b (str): The name of the second column to multiply.

    Returns:
        pd.DataFrame: A DataFrame containing the result of the element-wise multiplication of the two columns.
    """
    return features[col_a] * features[col_b]


def standard_scale_column(col: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard scaling to a single column in the input DataFrame.

    Args:
        col (pd.DataFrame): The input column to apply the standard scaling.

    Returns:
        pd.DataFrame: A DataFrame containing the standardized values of the input column.
    """
    mean = col.mean()
    std = col.std()
    return (col - mean) / std


def sqrt_transform_column(col: pd.DataFrame) -> pd.DataFrame:
    """
    Applies square root transformation to a single column in the input DataFrame.

    Args:
        col (pd.DataFrame): The input column to apply the square root transformation.

    Returns:
        pd.DataFrame: A DataFrame containing the square root-transformed values of the input column.
    """
    return col.apply(np.sqrt)


def save_features(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the given DataFrame to the specified file path as a CSV file.

    :param data: pd.DataFrame, the dataset to save
    :param file_path: str, path to save the dataset as a CSV file
    """
    logger.info("Attempting to save features to path: %s", file_path)

    try:
        # Save the DataFrame as a CSV file
        data.to_csv(file_path, index=False)
    except FileNotFoundError as e:
        logger.error("The specified directory does not exist: %s", str(e))
    except PermissionError as e:
        logger.error(
            "Permission denied when trying to save the features to %s: %s",
            file_path,
            str(e),
        )
    except pd.errors.EmptyDataError as e:
        logger.error(
            "The DataFrame is empty and cannot be saved as a CSV file: %s", str(e)
        )
    else:
        logger.info("Features saved successfully to file: %s", file_path)
