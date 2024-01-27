"""
This module provides various preprocessing methods.
"""

from enum import Enum
from pandas import DataFrame
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def drop_ft2(df: DataFrame):
    """
    Drops column "feature_2" from the dataframe.
    """
    return df.drop(columns="feature_2")


class DetectionMethod(Enum):
    """
    The detection method used for outlier detection.

    ISO_FOREST = IsolationForest from sklearn

    LOCAL_OUTLIER_FACTOR = LocalOutlierFactor from sklearn

    IQR = Using 1.5 * IQR method for calculating upper/lower limit
    """

    ISO_FOREST = 0
    LOCAL_OUTLIER_FACTOR = 1
    IQR = 2


class HandlingMethod(Enum):
    """
    The handling method used for dealing with outliers.

    REMOVE = Remove the instance completely

    CAP_AT_MIN_MAX = Cap the feature values at the border between inliers/outliers.
    """

    REMOVE = 0
    CAP_AT_MIN_MAX = 1


def remove_outliers(
    df: DataFrame,
    detection_method: DetectionMethod = DetectionMethod.ISO_FOREST,
    handling_method: HandlingMethod = HandlingMethod.REMOVE,
    random_state: int | None = None,
):
    """
    The function name is a bit misleading. It does not only remove outliers, but generally performs given handling method on the outliers after detecting them using given detection method.
    """

    # check detection method and obtain mask
    if detection_method == DetectionMethod.ISO_FOREST:
        mask = _get_iso_f_mask(df, random_state)
    elif detection_method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
        mask = _get_local_outlier_factor_mask(df)
    elif detection_method == DetectionMethod.IQR:
        mask = _get_iqr_mask(df)
    else:
        raise ValueError("Unknown DetectionMethod")

    # apply given handling method using mask
    if handling_method == HandlingMethod.REMOVE:
        return df[mask == 1]
    elif handling_method == HandlingMethod.CAP_AT_MIN_MAX:
        df_without_outliers = df[mask == 1]
        df_result = df.copy()
        for c in df.columns:
            cap_min = df_without_outliers[c].min()
            cap_max = df_without_outliers[c].max()
            clipped = df[c].clip(lower=cap_min, upper=cap_max)
            df_result.loc[mask == -1, c] = clipped

        return df_result
    else:
        raise ValueError("Unknown HandlingMethod")


def _get_iso_f_mask(df: DataFrame, random_state: int | None = None):
    iso_f = IsolationForest(random_state=random_state)
    return iso_f.fit_predict(df)


def _get_local_outlier_factor_mask(df: DataFrame):
    lof = LocalOutlierFactor()
    return lof.fit_predict(df)


def _get_iqr_mask(df: DataFrame):
    mask = DataFrame(True, index=df.index, columns=df.columns)

    # calculate 1.5*iqr mask for each column
    for col in df.columns:
        col_df = df[col]
        q1 = np.percentile(col_df, 25)
        q3 = np.percentile(col_df, 75)
        iqr = q3 - q1
        upper_lim = q3 + 1.5 * iqr
        lower_lim = q1 - 1.5 * iqr

        # apply mask on current column
        mask[col] &= (col_df > lower_lim) & (col_df < upper_lim)

    # combine mask of all columns (essential bitwise and along the columns)
    return mask.all(axis=1)
