from Utils.MultiStepDeal import *
import numpy as np
import pandas as pd
import torch
from functools import reduce

def MyErrorDistDeal(collect, TrainDist_df, eps, state, windows=None):

    if windows is None:
        windows = [(21, 0), (11,18)]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="inner"), collect)
    out_df = apply_preemptive_rules_multi(merged_df, windows=windows, up_th=2, down_th=10, apply_down=True, priority="up")
    out_df["Error"] = out_df["true_0"] - out_df["t_0"]
    out_df["Error_after"] = out_df["true_0"] - out_df["m_after"]

    if state:
        global_mu, global_sigma, hour_param = fit_residual_by_hour(TrainDist_df, shrink_strength=200)
        mu_pred = out_df["m_after"].to_numpy(dtype=float)
        r = out_df["Pred_Alpha"].to_numpy(dtype=float)
        p = r / (r + mu_pred)

        hour_array = out_df["Hour"].to_numpy(dtype=int)

        m_star, ft_star = find_m_vectorized_gh_hour(
            r_array=r,
            p_array=p,
            hour_array=hour_array,
            hour_param=hour_param,
            global_mu=global_mu,
            global_sigma=global_sigma,
            eps=eps,
            m_max=400,
            K=25
        )

        hour_table = (pd.DataFrame([(h, v[0], v[1], v[2]) for h, v in hour_param.items()],
                         columns=["Hour", "mu_err", "sigma_err", "n"]).sort_values("Hour"))
        #print(hour_table)

        out_df["m_final"] = m_star
        out_df["Error_final"] = out_df["true_0"] - out_df["m_final"]

    return out_df