# Import required modules
import logging
import pandas as pd
import numpy as np

# Set up logging configuration
logger = logging.getLogger("clouds")


def create_dataset(file_path: str, col_names: dict) -> pd.DataFrame:
    """Reads raw data from a file, processes it for two cloud classes, concatenates the results,
    and returns the combined data.

    Args:
        file_path (str): The path of the file containing the raw data.
        col_names (dict): A dictionary specifying the names of the columns in the resulting DataFrame.
    Returns:
        pd.DataFrame: A Pandas DataFrame containing the combined data.

    Raises:
        KeyError: If the "features" key is missing from the col_names dictionary.
        ValueError: If there is an error processing either cloud class data.
    """
    logger.debug("Attempting to create dataset")

    # Get column names from dictionary
    try:
        col_names = col_names["features"]
    except KeyError as e:
        logger.error("Error occurred while getting column names: %s", e)
        raise

    # Convert raw data to dataframe
    raw_data = read_raw_data(file_path)
    data = [[s for s in line.split(" ") if s != ""] for line in raw_data]
    first_cloud = process_cloud_class_data(data, 53, 1077, col_names, 0)
    second_cloud = process_cloud_class_data(data, 1082, 2105, col_names, 1)
    df = concatenate_dataframes(first_cloud, second_cloud)

    return df


def read_raw_data(file_path: str) -> list:
    """Reads raw data from a file and returns it as a list of strings.

    Args:
        file_path (str): The path of the file to read.

    Returns:
        list: A list of strings, where each string is a line from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    logger.debug("Trying to read raw data from file at %s", file_path)
    try:
        with open(file_path, "r") as file:
            raw_data = file.readlines()
    except FileNotFoundError as e:
        logger.error("Error occurred while reading raw data file: %s", e)
        raise
    return raw_data


def process_cloud_class_data(
    data: list, start: int, end: int, col_names: list, cloud_class: int
) -> pd.DataFrame:
    """Processes raw data for a cloud class and returns it as a Pandas DataFrame.

    Args:
        data (list): The raw data to process, as a list of strings.
        start (int): The line number to start processing data from.
        end (int): The line number to stop processing data at.
        col_names (list): The names of the columns in the resulting DataFrame.
        cloud_class (int): The class label for the cloud class being processed.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the processed data.

    Raises:
        ValueError: If the processed data is empty or contains invalid values.
    """
    logger.debug("Attempting to extract data from API response")
    try:
        cloud_data = data[start:end]
        cloud_data = [
            [float(s.replace("/n", "")) for s in cloud] for cloud in cloud_data
        ]
        cloud_data = pd.DataFrame(cloud_data, columns=col_names)
        cloud_data["class"] = np.full(len(cloud_data), cloud_class)
        logger.info("Cloud class %s data processed successfully", cloud_class)
    except (ValueError, IndexError) as e:
        logger.error(
            "Error occurred while processing cloud class %s data: %s", cloud_class, e
        )
        cloud_data = None
    return cloud_data


def concatenate_dataframes(
    first_cloud: pd.DataFrame, second_cloud: pd.DataFrame
) -> pd.DataFrame:
    """Concatenates two Pandas DataFrames and returns the result.

    Args:
        first_cloud (pd.DataFrame): The first DataFrame to concatenate.
        second_cloud (pd.DataFrame): The second DataFrame to concatenate.

    Returns:
        pd.DataFrame: A new DataFrame containing the concatenated data.

    Raises:
        ValueError: If both input DataFrames are empty.
    """
    logger.debug("Concatenating dataframes")
    if first_cloud is None and second_cloud is None:
        logger.error("Both cloud classes failed to process")
        raise ValueError("Both cloud classes failed to process")
    df = pd.concat([first_cloud, second_cloud], ignore_index=True)
    logger.info("DataFrames concatenated successfully")
    return df


def save_dataset(data: pd.DataFrame, file_path: str) -> None:
    """Saves a Pandas DataFrame to a CSV file.
    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path of the file to save the DataFrame to.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        PermissionError: If there is a permission issue when writing to the file.
        IOError: If there is an issue during the file writing process.
    """
    logger.info("Attempting to save dataset to path: %s", file_path)
    try:
        data.to_csv(file_path, index=False)
        logger.info("Dataset saved successfully to %s", file_path)
    except FileNotFoundError as e:
        logger.error("Error occurred while saving dataset: %s", e)
    except PermissionError as e:
        logger.error("Error occurred while saving dataset: %s", e)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.error("Error occurred while saving dataset: %s", e)
