"""
Two-phase calibration of the B-Bicycle digital twin.

Phase 1 (offline). A stacking ensemble learns the discrepancy between the DES
output and observed field revenue:

    y_field(x) = eta(x, theta) + delta(x) + eps          (Kennedy-O'Hagan form)

Rather than inferring theta by Bayesian inference, the simulator output
eta(x, theta) is supplied as a feature alongside 19 engineered decision-variable
features, and the ensemble learns the correction implicitly. Random Forest and
Gradient Boosting are fitted on the scaled feature matrix; a Bayesian Ridge
layer combines their predictions with the raw simulator output.

Phase 2 (online). A two-parameter recursive Bayesian correction

    y_hat_t = scale_t * y_phase1_t + bias_t

is updated month by month across the test horizon, weighting each observation
by the inverse of its reported variance. Only the earlier observations
influence any given prediction, so there is no look-ahead.

Note on imports: ``cross_val_score`` and the three metric functions were
implicit in the original notebook, where every cell shared one namespace. They
are imported explicitly here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import create_features

RANDOM_STATE = 42


def smape(y_true, y_pred, eps=1e-9):
    """Symmetric MAPE (%), robust to observations near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def mape(y_true, y_pred):
    """Plain MAPE (%), computed over non-zero observations only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nz = y_true != 0
    return float(np.mean(np.abs((y_pred[nz] - y_true[nz]) / y_true[nz])) * 100.0)


class EnsembleCalibrator:
    """RF + GBR + BayesianRidge stack, plus an online bias/scale corrector."""

    def __init__(self):
        self.rf = None
        self.gbr = None
        self.bayesian_ridge = None
        self.scaler = StandardScaler()
        self.sim_scaler = StandardScaler()
        self.online_bias_history = [0.0]
        self.online_scale_history = [1.0]
        self.feat_columns = []

    # -- Phase 1 -----------------------------------------------------------

    def calibrate_phase1(self, df_train, sim_preds, verbose=True):
        """Fit the stacking ensemble on the training months."""
        y_field = df_train['NetRevenue'].values
        y_sim = np.asarray(sim_preds, dtype=float)

        feat = create_features(df_train)
        feat['sim_pred'] = y_sim
        X_scaled = self.scaler.fit_transform(feat.values)

        if verbose:
            print("\n  Phase 1: Ensemble Surrogate Calibration (KOH framework)")
            print("  " + "-" * 55)
            rmse_sim = np.sqrt(mean_squared_error(y_field, y_sim))
            print(f"  Uncalibrated sim train RMSE: ${rmse_sim:,.0f} "
                  f"(R2={r2_score(y_field, y_sim):.4f})")
            print("  Training Random Forest ...")

        self.rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
        self.rf.fit(X_scaled, y_field)
        rf_pred = self.rf.predict(X_scaled)

        if verbose:
            print("  Training Gradient Boosting ...")
        self.gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE)
        self.gbr.fit(X_scaled, y_field)
        gbr_pred = self.gbr.predict(X_scaled)

        stack_X = np.column_stack([rf_pred, gbr_pred, y_sim])
        self.bayesian_ridge = BayesianRidge(max_iter=500)
        self.bayesian_ridge.fit(stack_X, y_field)
        final_pred = self.bayesian_ridge.predict(stack_X)

        # 5-fold CV on the RF alone -- the honest generalization estimate.
        # The train RMSE below is fitted-on-itself and is NOT a performance
        # figure; quote cv_rmse or the held-out test RMSE instead.
        cv = cross_val_score(
            RandomForestRegressor(n_estimators=200, max_depth=12,
                                  min_samples_leaf=3, random_state=RANDOM_STATE),
            X_scaled, y_field, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = float(-cv.mean())

        train_rmse = float(np.sqrt(mean_squared_error(y_field, final_pred)))
        train_r2 = float(r2_score(y_field, final_pred))

        self.feat_columns = list(feat.columns)
        importances = self.rf.feature_importances_
        importance = {n: float(v) for n, v in zip(self.feat_columns, importances)}

        if verbose:
            print(f"\n  Ensemble train RMSE: ${train_rmse:,.0f} (R2={train_r2:.4f})"
                  "   <- in-sample, not a performance figure")
            print(f"  5-fold CV RMSE:      ${cv_rmse:,.0f}   <- generalization estimate")
            top = sorted(importance.items(), key=lambda kv: -kv[1])[:6]
            print("  Top features: " + ", ".join(f"{k}({v:.3f})" for k, v in top))

        return {
            'train_pred': final_pred,
            'cv_rmse': cv_rmse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'feature_importance': importance,
        }

    # -- prediction --------------------------------------------------------

    def _stack_inputs(self, df, sim_preds):
        feat = create_features(df)
        feat['sim_pred'] = np.asarray(sim_preds, dtype=float)
        X = self.scaler.transform(feat.values)
        rf_p = self.rf.predict(X)
        gbr_p = self.gbr.predict(X)
        return X, np.column_stack([rf_p, gbr_p, np.asarray(sim_preds, dtype=float)])

    def predict(self, df, sim_preds):
        _, stack = self._stack_inputs(df, sim_preds)
        return self.bayesian_ridge.predict(stack)

    def predict_uncertainty(self, df, sim_preds):
        """Point prediction plus 1-sigma, combining ridge and RF-tree spread."""
        X, stack = self._stack_inputs(df, sim_preds)
        mean, std_br = self.bayesian_ridge.predict(stack, return_std=True)
        tree_preds = np.array([t.predict(X) for t in self.rf.estimators_])
        rf_std = np.std(tree_preds, axis=0)
        return mean, np.sqrt(std_br ** 2 + rf_std ** 2)

    # -- Phase 2 -----------------------------------------------------------

    def online_update(self, df_test, sim_preds_test, verbose=True):
        """Recursive Bayesian bias/scale correction across the test horizon.

        Returns a TUPLE ``(phase1_preds, corrected_preds)``. Unpack it --
        assigning the tuple to a single name produces a shape error downstream.
        """
        if verbose:
            print("\n  Phase 2: Online Recursive Bayesian Updating")
            print("  " + "-" * 55)

        bias_mu, bias_prec = 0.0, 1e-14
        scale_mu, scale_prec = 1.0, 1e-12

        preds_p1, preds_online = [], []
        self.online_bias_history = [0.0]
        self.online_scale_history = [1.0]

        for i, (_, row) in enumerate(df_test.iterrows()):
            p1 = self.predict(pd.DataFrame([row]), np.array([sim_preds_test[i]]))[0]
            preds_p1.append(p1)

            corrected = scale_mu * p1 + bias_mu
            preds_online.append(corrected)

            # Observe, then update -- never the other way round.
            y_obs = row['NetRevenue']
            obs_noise_var = max(row['Standard Deviation'] ** 2, 1e-6)
            noise_prec = 1.0 / obs_noise_var

            residual = y_obs - scale_mu * p1
            new_bp = bias_prec + noise_prec
            bias_mu = (bias_prec * bias_mu + noise_prec * residual) / new_bp
            bias_prec = new_bp

            if abs(p1) > 1e-4:
                obs_scale = (y_obs - bias_mu) / p1
                new_sp = scale_prec + noise_prec * p1 ** 2
                scale_mu = (scale_prec * scale_mu
                            + noise_prec * p1 ** 2 * obs_scale) / new_sp
                scale_prec = new_sp
                scale_mu = float(np.clip(scale_mu, 0.3, 3.0))

            self.online_bias_history.append(bias_mu)
            self.online_scale_history.append(scale_mu)

            if verbose and i % 12 == 0:
                print(f"    Month {i:3d}: pred=${corrected:>12,.0f} "
                      f"obs=${y_obs:>12,.0f}  bias=${bias_mu:>10,.0f} "
                      f"scale={scale_mu:.3f}")

        return np.array(preds_p1), np.array(preds_online)


def evaluate(y_true, y_pred) -> dict:
    """RMSE / MAE / MAPE / sMAPE / R2 for one prediction vector."""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'r2': float(r2_score(y_true, y_pred)),
    }
