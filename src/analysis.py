# Import required modules
from pathlib import Path
import logging
import warnings
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging configuration
logger = logging.getLogger("clouds")


def save_figures(
    features: pd.DataFrame, directory: Path, analysis_config: dict
) -> None:
    """
    Generate and save histograms for each feature in the given DataFrame, using the provided colors and directory.

    Parameters:
    - features: pd.DataFrame, the DataFrame containing the features to plot
    - directory: Path, the directory to save the generated figures to
    - analysis_config: dict, the configuration parameters to use for the analysis (including colors, etc.)

    Returns:
    - None

    """
    # Get the colors from the analysis config
    colors = analysis_config["colors"]
    logger.debug("Plot colors: %s", colors)

    # Log the start of the function
    logger.info(
        "Generating and saving histograms for %d features to directory: %s",
        len(features.columns),
        directory,
    )

    for feat in features.columns:
        logger.debug("Processing feature: %s", feat)
        # Catch any warnings that may occur during figure generation and saving
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Apply any warnings filter from the analysis config
            apply_warnings_filter(analysis_config["log_warnings"])

            try:
                # Create the figure and plot the histogram
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_histogram(ax, features, feat, colors)
            except ValueError as e:
                logger.error(
                    "Error while creating plot for feature %s: %s", feat, str(e)
                )
                continue

            try:
                # Save the figure to the specified directory
                fig_path = directory / f"{feat}.png"
                logger.debug("Figure path: %s", fig_path)
                fig.savefig(fig_path, bbox_inches="tight")
                logger.info("Figure for feature %s saved at %s", feat, fig_path)
            except (FileNotFoundError, PermissionError) as e:
                logger.error(
                    "Error while saving figure for feature %s to %s: %s",
                    feat,
                    fig_path,
                    str(e),
                )
                continue

            # If there were any warnings during figure generation and saving, log them
            if caught_warnings:
                for warning in caught_warnings:
                    logger.warning(
                        "Warning while generating and saving figure for feature %s: %s",
                        feat,
                        str(warning.message),
                    )


def apply_warnings_filter(log_warnings: bool):
    """
    Apply a warnings filter based on the provided boolean value.

    Parameters:
    - log_warnings: bool, whether to log warnings or ignore them

    Returns:
    - None

    """
    logger.debug(
        "Applying warnings filter with log_warnings=%s", log_warnings
    )  # Added DEBUG log

    # If log_warnings is True, log all warnings; otherwise, ignore them
    if log_warnings:
        warnings.simplefilter("always")
    else:
        warnings.simplefilter("ignore")


def plot_histogram(ax, features, feat, colors):
    """
    Plot a histogram for the given feature, using the provided colors.

    Parameters:
    - ax: matplotlib.axes.Axes, the axes to plot on
    - features: pd.DataFrame, the DataFrame containing the feature data
    - feat: str, the name of the feature to plot
    - colors: list of str, the colors to use for the histogram bars

    Returns:
    - None

    """
    logger.debug("Plotting histogram for feature: %s", feat)

    # Plot the histogram using the specified colors
    ax.hist(
        [
            features[features["class"] == 0][feat].values,
            features[features["class"] == 1][feat].values,
        ],
        color=colors,
        label=["Class 0", "Class 1"],
    )
    # Set the x and y axis labels and the legend
    ax.set_xlabel(" ".join(feat.split("_")).capitalize())
    ax.set_ylabel("Number of observations")
    ax.legend(loc="upper right")
