"""
Feature engineering for the calibration ensemble.

Nineteen features are built from the six decision variables. The calibrator
appends the simulator output as a twentieth column (``sim_pred``) before
scaling, so the fitted StandardScaler expects exactly 20 columns in the order
produced here. Do not reorder, rename, or insert columns: the committed
``p1_artifacts/calibrator.joblib`` was fitted against this exact layout.
"""

import numpy as np
import pandas as pd

DECISION_VARS = ['C', 'R', 'Q', 'W', 'M1', 'M2']

FEATURE_COLUMNS = [
    'C', 'R', 'Q', 'W', 'M1', 'M2',
    'C_x_W', 'R_x_Q', 'M1_x_M2', 'M_total', 'M_invest',
    'W_inv', 'R_over_Q', 'Q_x_W', 'C_x_M1', 'W_sq',
    'logW', 'logR', 'logQ',
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 19 engineered features from the six decision variables."""
    X = df[DECISION_VARS].values.astype(float)
    feat = pd.DataFrame(X, columns=DECISION_VARS)

    feat['C_x_W'] = feat['C'] * feat['W']            # contract x WIP limit
    feat['R_x_Q'] = feat['R'] * feat['Q']            # inventory policy scale
    feat['M1_x_M2'] = feat['M1'] * feat['M2']        # capacity synergy
    feat['M_total'] = feat['M1'] + feat['M2']        # total machines
    feat['M_invest'] = 50000 * feat['M_total']       # monthly machine spend
    feat['W_inv'] = 1.0 / (feat['W'] + 1)            # diminishing WIP effect
    feat['R_over_Q'] = feat['R'] / (feat['Q'] + 1)   # replenishment frequency
    feat['Q_x_W'] = feat['Q'] * feat['W']            # batch x WIP
    feat['C_x_M1'] = feat['C'] * feat['M1']          # contract x assembly
    feat['W_sq'] = feat['W'] ** 2                    # quadratic WIP
    feat['logW'] = np.log1p(feat['W'])
    feat['logR'] = np.log1p(feat['R'])
    feat['logQ'] = np.log1p(feat['Q'])

    return feat
