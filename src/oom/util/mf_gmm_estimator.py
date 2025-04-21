from typing import Callable, Optional

from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import scipy as sp


def l2_distance_gmm(weights_true, pdfs_true, weights_gmm, pdfs_gmm):
    """
    Compute the squared L2 distance between two Gaussian Mixture Models (GMMs).

    Parameters:
        weights_true (np.array): Coefficients of the true GMM.
        pdfs_true (list of scipy.stats.norm): List of norm objects for the true GMM.
        weights_gmm (np.array): Coefficients of the learned GMM.
        pdfs_gmm (list of scipy.stats.norm): List of norm objects for the learned GMM.

    Returns:
        float: Squared L2 distance between the two GMMs.
    """
    # Helper function to compute the integral of the product of two Gaussians
    def gaussian_product_integral(mu1, sigma1, mu2, sigma2):
        sigma_sum = sigma1 + sigma2
        return sp.stats.norm.pdf(mu1, loc=mu2, scale=np.sqrt(sigma_sum))

    # Compute the self-interaction term for the true GMM
    true_self_term = 0.0
    for i in range(len(weights_true)):
        for j in range(len(weights_true)):
            true_self_term += weights_true[i] * weights_true[j] * gaussian_product_integral(
                pdfs_true[i].mean(), pdfs_true[i].std()**2,
                pdfs_true[j].mean(), pdfs_true[j].std()**2
            )

    # Compute the self-interaction term for the learned GMM
    gmm_self_term = 0.0
    for i in range(len(weights_gmm)):
        for j in range(len(weights_gmm)):
            gmm_self_term += weights_gmm[i] * weights_gmm[j] * gaussian_product_integral(
                pdfs_gmm[i].mean(), pdfs_gmm[i].std()**2,
                pdfs_gmm[j].mean(), pdfs_gmm[j].std()**2
            )

    # Compute the cross-interaction term between the true and learned GMMs
    cross_term = 0.0
    for i in range(len(weights_true)):
        for j in range(len(weights_gmm)):
            cross_term += weights_true[i] * weights_gmm[j] * gaussian_product_integral(
                pdfs_true[i].mean(), pdfs_true[i].std()**2,
                pdfs_gmm[j].mean(), pdfs_gmm[j].std()**2
            )

    # Compute the squared L2 distance
    l2_squared = true_self_term + gmm_self_term - 2 * cross_term
    return l2_squared


def get_from_gmm(
    data: np.array,
	n_components_range: list,
    ret_metrics: bool = True,
    true_pdfs: Optional[list[Callable]] = None,
    true_seqweights: Optional[list[float]] = None
) -> list[Callable] | tuple[list[Callable], dict[str, float]]:
    """
    Models input data using a Gaussian Mixture model. This function internally performs
    cross-validation using 90% of data as train and 10% as validation. It will report the
    optimal membership function estimates, chosen based on the validation NLL. Optionally,
    it can report the NLL metric and even the L2 distance to another (target) GMM.

    Args:
        data: the array of observation data to use in the clustering task
        ret_metrics: whether to return metrics together with the membership function estimates
        true_pdfs: the true PDFs expected to be modeled by the GMM (if this parameter and
            `true_seqweights` are both provided, the L2 distance between GMMs will also be
            computed)
        true_seqweights: the weighing of each true PDF in the Gaussian mixture which is expected
            to be modeled by this GMM (if this parameter and `true_pdfs` are both provided, the
            L2 distance between GMMs will also be computed)

    Returns:
        A Python list of scipy.stats.rv_continuous objects representing the Gaussians composing
        the best GMM after cross-validation. If ret_metrics is true, a dictionary of metric names
        and metric values containing the best GMM's evaluation is also returned, resulting in a
        tuple (list, dict)
    """
    length = len(data)
    
    val_size = int(0.1 * length)
    train_data = data[ : -val_size]
    val_data = data[-val_size : ]
    
    metrics = {"est_alphabet_size": [], "nll_gmm": []}
    
    for est_alphabet_size in n_components_range:
        # Fit a GMM on the (split) train set
        gmm = GaussianMixture(
            n_components = est_alphabet_size,
            covariance_type = "full",
            init_params = "random_from_data",
            max_iter = 1000,
            tol = 1e-5,
            random_state = 0
        )
        gmm.fit(train_data)

        # Assess GMM on validation data and record result
        nll_here = -gmm.score(val_data)
        metrics["est_alphabet_size"].append(est_alphabet_size)
        metrics["nll_gmm"].append(nll_here)

    # Sort by NLL and get alphabet size for lowest NLL
    metrics_df = pd.DataFrame(metrics).sort_values("nll_gmm", ascending = True)
    best_alphabet_size = metrics_df["est_alphabet_size"].iloc[0]
    best_nll = metrics_df["nll_gmm"].iloc[0]

    # Fit GMM using entire training sequence (without validation split)
    best_gmm = GaussianMixture(
        n_components = best_alphabet_size,
        covariance_type = "full",
        init_params = "random_from_data",
        max_iter = 1000,
        tol = 1e-5,
        random_state = 0
    )
    best_gmm.fit(data)

    # Get PDFs from this best GMM
    means_gmm = best_gmm.means_
    covs_gmm  = best_gmm.covariances_
    weights_gmm = best_gmm.weights_
    
    pdfs_best_gmm = []
    for idx in range(best_alphabet_size):
        pdf = sp.stats.norm(loc = means_gmm[idx][0], scale = np.sqrt(covs_gmm[idx][0][0]))
        pdfs_best_gmm.append(pdf)

    results_metrics = {"alphabet_size": best_alphabet_size, "nll_gmm": best_nll}
    
    if true_pdfs is not None and true_seqweights is not None:
        l2_here = l2_distance_gmm(true_seqweights, true_pdfs, weights_gmm, pdfs_best_gmm)
        results_metrics["l2_gmm"] = l2_here

    if ret_metrics:
        return pdfs_best_gmm, results_metrics
    return pdfs_best_gmm