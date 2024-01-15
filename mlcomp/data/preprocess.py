from enum import Enum
from pandas import DataFrame
from sklearn.ensemble import IsolationForest


def drop_ft2(df: DataFrame):
    return df.drop(columns="feature_2")


class DetectionMethod(Enum):
    ISO_FOREST = 0


class HandlingMethod(Enum):
    REMOVE = 0
    CAP_AT_MIN_MAX = 1


def remove_outliers(
    df: DataFrame,
    detection_method: DetectionMethod = DetectionMethod.ISO_FOREST,
    handling_method: HandlingMethod = HandlingMethod.REMOVE,
    random_state: int | None = None
):
    if detection_method == DetectionMethod.ISO_FOREST:
        mask = _get_iso_f_mask(df, random_state)
    else:
        raise ValueError("Unknown DetectionMethod")

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

