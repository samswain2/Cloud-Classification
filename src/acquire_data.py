# Import required modules
import logging
import sys
import time
from pathlib import Path
import requests

# Set up logging configuration
logger = logging.getLogger("clouds")


def get_data(
    url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2
) -> bytes:
    """Acquire data from a URL with retry mechanism.

    Args:
        url (str): The URL to fetch data from.
        attempts (int): The number of attempts to fetch data before giving up. Default is 4.
        wait (int): The base wait time (in seconds) between attempts. Default is 3.
        wait_multiple (int): The multiplier to increase the wait time for each retry. Default is 2.

    Returns:
        bytes: The data fetched from the URL.

    Raises:
        requests.exceptions.RequestException: If fetching the data fails after all attempts.
    """

    logger.info("Fetching data from %s", url)

    # Loop through a number of attempts to fetch data
    for num_tries in range(1, attempts + 1):
        logger.debug("Attempt %d to fetch data from %s", num_tries, url)
        try:
            # Make a request to the URL with a timeout that increases with each attempt
            response = requests.get(url, timeout=wait * num_tries * wait_multiple)
            logger.debug("Response status code: %d", response.status_code)
            data = response.text
            return data

        except requests.exceptions.RequestException as e:
            # If the request fails, log a warning and sleep for a period of time before trying again
            logger.warning(
                "Attempt %s: Error while trying to get data from %s: %s",
                num_tries,
                url,
                e,
            )
            time.sleep(wait * wait_multiple * num_tries)

    # If all attempts fail, raise an exception
    raise requests.exceptions.RequestException(
        f"Failed to fetch data from {url} after {attempts} attempts"
    )


def write_data(data: str, save_path: Path) -> None:
    """Write data to a file.

    Args:
        data (bytes): The data to be written to the file.
        save_path (Path): The file path to write the data to.
    """
    logger.info("Writing data to %s", save_path)

    try:
        # Write the data to the specified file path
        logger.debug("Save path: %s", save_path)
        with save_path.open("w") as f:
            f.write(data)
        logger.info("Wrote data to %s", save_path)
    except OSError as e:
        logger.error("Error while writing data to %s: %s", save_path, e)
        logger.info("Continuing with the program.")


def acquire_data(url: str, save_path: Path) -> None:
    """Acquire data from a URL and write it to a local file.

    Args:
        url (str): The URL for where the data to be acquired is stored.
        save_path (Path): The local path to write data to.
    """
    logger.info("Starting data acquisition")

    # Fetch data from the URL
    url_contents = get_data(url)

    logger.info("Completed data acquisition")
    logger.debug("Attempting to save url contents to path: %s", save_path)

    try:
        # Write the data to the specified file path
        write_data(url_contents, save_path)
        logger.info("Data written to %s", save_path)
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save dataset to.")
        sys.exit(1)

    logger.info("Data acquisition complete")
