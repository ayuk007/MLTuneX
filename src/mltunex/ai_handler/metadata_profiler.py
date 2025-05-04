"""
Metadata Profiler module for MLTuneX.

This module provides functionality for extracting and analyzing metadata from
datasets. It supports both preprocessed and raw data, offering insights into
data distributions, correlations, and statistical properties.

Examples:
    >>> profiler = MetaDataProfiler(data, "target", "classification")
    >>> insights = profiler.extract()
"""

from typing import List, TypedDict, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np


class MetaDataProfiler:
    """
    Data profiler for extracting dataset metadata and statistics.

    This class analyzes datasets to extract various statistical properties,
    feature distributions, and relationships. It supports both split and
    combined dataset formats.

    Parameters
    ----------
    data : Union[Tuple, Any]
        Either a tuple of (x_train, x_test, y_train, y_test) or a DataFrame.
    target_column : str
        Name of the target variable column.
    task_type : str
        Type of ML task ('classification' or 'regression').
    is_preprocessed : bool, optional (default=False)
        Whether the data has been preprocessed.

    Attributes
    ----------
    data : pd.DataFrame
        Combined dataset for analysis.
    target_column : str
        Name of target variable.
    task_type : str
        Type of ML task.
    is_preprocessed : bool
        Preprocessing status flag.
    """

    def __init__(self, data: Union[Tuple, Any], target_column: str, 
                 task_type: str, is_preprocessed: bool = False):
        """Initialize MetaDataProfiler with dataset and configuration."""
        self.target_column = target_column
        self.is_preprocessed = is_preprocessed
        self.task_type = task_type
        self.data = self.merge_data_split(data)

    def get_shape(self) -> str:
        """
        Get dataset dimensions.

        Returns
        -------
        str
            JSON string with number of rows and features.
        """
        return str({"num_rows": self.data.shape[0], "num_features": self.data.shape[1]})

    def find_missing_values(self) -> pd.Series:
        """
        Count missing values in each column.

        Returns
        -------
        pd.Series
            Count of null values per column.
        """
        return self.data.isnull().sum()

    def get_data_stats(self) -> str:
        """
        Get basic statistical description of the dataset.

        Returns
        -------
        str
            JSON string containing dataset statistics.
        """
        return self.data.describe().to_json()

    def feature_distribution_insights(self) -> str:
        """
        Analyze feature distributions based on variance.

        Groups features into categories based on their variance levels:
        - Dead/Constant: Zero variance
        - Low Variance: Bottom 5%
        - Medium Variance: Middle 90%
        - High Variance: Top 5%

        Returns
        -------
        str
            JSON string with grouped feature names.
        """
        stds = self.data.std()
        # Calculate variance thresholds
        low_threshold = stds.quantile(0.05)   # Bottom 5% = low variance
        high_threshold = stds.quantile(0.95)  # Top 5% = high variance
        
        # Group features by variance levels
        grouped_features = {
            'Dead/Constant': stds[stds == 0].index.tolist(),
            'Low Variance': stds[(stds > 0) & (stds <= low_threshold)].index.tolist(),
            'Medium Variance': stds[(stds > low_threshold) & (stds <= high_threshold)].index.tolist(),
            'High Variance': stds[stds > high_threshold].index.tolist()
        }
        return str(grouped_features)

    def correlation_analysis(self) -> str:
        """
        Calculate correlation matrix for all features.

        Returns
        -------
        str
            JSON string containing Pearson correlation matrix.
        """
        corr_matrix = self.data.corr(method='pearson')
        return corr_matrix.to_json()

    def get_data_types(self) -> str:
        """
        Get data types of all columns.

        Returns
        -------
        str
            JSON string with column data types.
        """
        return self.data.dtypes.astype(str).to_json()

    def target_distribution(self) -> str:
        """
        Analyze target variable distribution.

        Returns
        -------
        str
            JSON string with normalized value counts.
        """
        return self.data[self.target_column].value_counts(normalize=True).to_json()

    def get_skew_kurtosis(self) -> str:
        """
        Calculate skewness and kurtosis for all features.

        Returns
        -------
        str
            JSON string containing skewness and kurtosis values.
        """
        return str({
            "skewness": self.data.skew().to_dict(),
            "kurtosis": self.data.kurt().to_dict()
        })

    def get_features(self) -> str:
        """
        Get list of feature names excluding target.

        Returns
        -------
        str
            String representation of feature list.
        """
        return str([col for col in self.data.columns if col != self.target_column])

    def extract(self) -> str:
        """
        Extract complete dataset profile.

        Returns
        -------
        str
            XML-formatted string containing all profile information.
            Returns None if data is preprocessed.
        """
        if not self.is_preprocessed:
            data_insights = f"""
            <Features>{self.get_features()}</Features>
            <Target>{self.target_column}</Target>
            <TaskType>{self.task_type}</TaskType>
            <Shape>{self.get_shape()}</Shape>
            <DataStats>{self.get_data_stats()}</DataStats>
            <FeatureDistributionInsights>{self.feature_distribution_insights()}</FeatureDistributionInsights>
            <CorrelationAnalysis>{self.correlation_analysis()}</CorrelationAnalysis>
            """
            return data_insights
        return None

    def merge_data_split(self, data: Union[Tuple, Any]) -> pd.DataFrame:
        """
        Merge split dataset into single DataFrame.

        Parameters
        ----------
        data : Union[Tuple, Any]
            Either split data tuple or DataFrame.

        Returns
        -------
        pd.DataFrame
            Combined dataset.

        Raises
        ------
        ValueError
            If data format is not supported.
        """
        if isinstance(data, tuple):
            # Unpack and combine split datasets
            x_train, x_test, y_train, y_test = data
            columns = x_train.columns.tolist() + [self.target_column]
            x = np.concatenate((x_train, x_test), axis=0)
            y_train = y_train.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)
            y = np.concatenate((y_train, y_test), axis=0)
            data = np.concatenate((x, y), axis=1)
            data = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            pass
        else:
            raise ValueError("Data must be tuple with (x_train, x_test, y_train, y_test) or any DataFrame")

        return data