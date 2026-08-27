"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""
import multiprocessing
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import OUTPUT_TEMPLATE
from .metric_utils import compute_logits, compute_pairwise_cd, \
    compute_pairwise_emd, pcd2bev_sum, compute_pairwise_cd_batch, pcd2bev_bin
from .fid_score import calculate_frechet_distance


def evaluate(reference, samples, metrics, data):
    # perceptual
    if 'frid' in metrics:
        compute_frid(reference, samples, data)
    if 'fsvd' in metrics:
        compute_fsvd(reference, samples, data)
    if 'fpvd' in metrics:
        compute_fpvd(reference, samples, data)

    # reconstruction
    if 'cd' in metrics:
        compute_cd(reference, samples)
    if 'emd' in metrics:
        compute_emd(reference, samples)

    # statistical
    if 'jsd' in metrics:
        compute_jsd(reference, samples, data)
    if 'mmd' in metrics:
        compute_mmd(reference, samples, data)

    if 'edgermse' in metrics:
        compute_edge_rmse(reference, samples)
    if 'regularity' in metrics:
        compute_surface_regularity(reference, samples)


def compute_cd(reference, samples):
    """
    Calculate score of Chamfer Distance (CD)

    """
    print('Evaluating (CD) ...')
    results = []
    for x, y in tqdm(zip(reference, samples), total=len(reference)):
        d = compute_pairwise_cd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('CD  ', score))


def compute_emd(reference, samples):
    """
    Calculate score of Earth Mover's Distance (EMD)

    """
    print('Evaluating (EMD) ...')
    results = []
    for x, y in tqdm(zip(reference, samples), total=len(reference)):
        d = compute_pairwise_emd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('EMD ', score))


def compute_mmd(reference, samples, data, dist='cd', verbose=True):
    """
    Calculate the score of Minimum Matching Distance (MMD)

    """
    print('Evaluating (MMD) ...')
    assert dist in ['cd', 'emd']
    reference, samples = pcd2bev_bin(data, reference, samples)
    compute_dist_func = compute_pairwise_cd_batch if dist == 'cd' else compute_pairwise_emd
    results = []
    for r in tqdm(reference, disable=not verbose):
        dists = compute_dist_func(r, samples)
        results.append(min(dists))
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('MMD ', score))


def compute_jsd(reference, samples, data):
    """
    Calculate the score of Jensen-Shannon Divergence (JSD)

    """
    print('Evaluating (JSD) ...')
    reference, samples = pcd2bev_sum(data, reference, samples)
    reference = (reference / np.sum(reference)).flatten()
    samples = (samples / np.sum(samples)).flatten()
    score = jensenshannon(reference, samples)
    print(OUTPUT_TEMPLATE.format('JSD ', score))


def compute_fd(reference, samples):
    mu1, mu2 = np.mean(reference, axis=0), np.mean(samples, axis=0)
    sigma1, sigma2 = np.cov(reference, rowvar=False), np.cov(samples, rowvar=False)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return distance


def compute_frid(reference, samples, data):
    """
    Calculate the score of Fréchet Range Image Distance (FRID)

    """
    print('Evaluating (FRID) ...')
    gt_logits, samples_logits = compute_logits(data, 'range', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FRID', score))


def compute_fsvd(reference, samples, data):
    """
    Calculate the score of Fréchet Sparse Volume Distance (FSVD)

    """
    print('Evaluating (FSVD) ...')
    gt_logits, samples_logits = compute_logits(data, 'voxel', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FSVD', score))


def compute_fpvd(reference, samples, data):
    """
    Calculate the score of Fréchet Point-based Volume Distance (FPVD)

    """
    print('Evaluating (FPVD) ...')
    gt_logits, samples_logits = compute_logits(data, 'point_voxel', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FPVD', score))
    

def single_edgermse_rv_lidar(pred, gt, edge_thresh=0.5, eps=1e-6):
    """
    LiDAR-aware EdgeRMSE for RV images, masking zero (no-return) areas.

    Args:
        pred: Tensor [..., H, W] predicted depth/range
        gt:   Tensor [..., H, W] ground truth depth/range
        edge_thresh: depth gradient threshold
        eps: numerical stability

    Returns:
        EdgeRMSE scalar
    """
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    assert pred.shape == gt.shape, "Shape mismatch"

    pred = pred.float()
    gt = gt.float()

    # Valid return mask: GT > 0
    valid_mask = gt > 0

    # Compute finite differences on GT
    dx = gt[..., :, 1:] - gt[..., :, :-1]
    dy = gt[..., 1:, :] - gt[..., :-1, :]

    dx = F.pad(dx, (0, 1))
    dy = F.pad(dy, (0, 0, 0, 1))

    grad_mag = torch.sqrt(dx**2 + dy**2)

    # Edge mask: strong depth change AND valid returns
    neighbor_valid = F.pad(valid_mask[..., :, 1:], (0,1)) & valid_mask
    edge_mask = (grad_mag > edge_thresh) & neighbor_valid

    # Squared error only where GT valid
    se = (pred - gt) ** 2
    masked_se = se[edge_mask]

    if masked_se.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.sqrt(masked_se.mean() + eps).item()

def compute_edge_rmse(reference, samples):
    results = []
    for [gt, pred] in tqdm(zip(reference, samples), total=len(reference)):
        results.append(single_edgermse_rv_lidar(gt=gt, pred=pred, edge_thresh=0.5))
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('EdgeRMSE ', score))


def surface_regularity_pca(points, k=30, max_points=200000, seed=0):
    """
    Surface regularity metrics from local PCA neighborhoods.

    Args:
        points: (N, 3) numpy array
        k: number of neighbors for local plane fit (>= 10 recommended)
        max_points: subsample if N is huge (for speed)
        seed: RNG seed for subsampling

    Returns:
        dict with:
          - planarity_residual_mean
          - planarity_residual_median
          - surface_variation_mean
          - surface_variation_median
          - per_point_residual (M,)
          - per_point_surface_variation (M,)
    """
    pts = np.asarray(points, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 3, "points must be (N, 3)"

    # Remove NaN/Inf
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    N = pts.shape[0]
    if N < k + 1:
        raise ValueError(f"Not enough points: N={N}, k={k}")

    # Optional subsample for speed
    if N > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_points, replace=False)
        pts_eval = pts[idx]
    else:
        pts_eval = pts

    # kNN in the same point cloud
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(pts)
    _, nbr_idx = nn.kneighbors(pts_eval, return_distance=True)  # (M, k)

    residuals = np.empty((pts_eval.shape[0],), dtype=np.float64)
    surf_var = np.empty((pts_eval.shape[0],), dtype=np.float64)

    for i in range(pts_eval.shape[0]):
        nbrs = pts[nbr_idx[i]]  # (k,3)
        mu = nbrs.mean(axis=0)
        X = nbrs - mu

        # Covariance + eigen decomposition
        C = (X.T @ X) / max(k - 1, 1)
        # eigh returns ascending eigenvalues
        w, v = np.linalg.eigh(C)
        w = np.maximum(w, 0.0)  # numerical safety

        # Normal is eigenvector of smallest eigenvalue
        n = v[:, 0]
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm

        # Point-to-plane distance for neighbors
        d = np.abs((nbrs - mu) @ n)  # (k,)
        residuals[i] = d.mean()

        # Surface variation: λ3 / (λ1+λ2+λ3), with λ1<=λ2<=λ3 here
        denom = w.sum() + 1e-12
        surf_var[i] = w[0] / denom

    return {
        "planarity_residual_mean": float(residuals.mean()),
        "planarity_residual_median": float(np.median(residuals)),
        "surface_variation_mean": float(surf_var.mean()),
        "surface_variation_median": float(np.median(surf_var)),
        "per_point_residual": residuals,
        "per_point_surface_variation": surf_var,
    }


def surface_regularity_compare(pred_points, gt_points=None, k=30, max_points=200000):
    """
    Convenience wrapper to compute regularity of pred (and GT if provided),
    plus deltas that are easy to report in tables.
    """
    pred = surface_regularity_pca(pred_points, k=k, max_points=max_points, seed=0)

    out = {
        "pred_planarity_residual_mean": pred["planarity_residual_mean"],
        "pred_planarity_residual_median": pred["planarity_residual_median"],
        "pred_surface_variation_mean": pred["surface_variation_mean"],
        "pred_surface_variation_median": pred["surface_variation_median"],
    }

    if gt_points is not None:
        gt = surface_regularity_pca(gt_points, k=k, max_points=max_points, seed=1)
        out.update({
            "gt_planarity_residual_mean": gt["planarity_residual_mean"],
            "gt_planarity_residual_median": gt["planarity_residual_median"],
            "gt_surface_variation_mean": gt["surface_variation_mean"],
            "gt_surface_variation_median": gt["surface_variation_median"],

            "delta_planarity_residual_mean": out["pred_planarity_residual_mean"] - gt["planarity_residual_mean"],
            "delta_surface_variation_mean": out["pred_surface_variation_mean"] - gt["surface_variation_mean"],
        })

    return out


def compute_surface_regularity(reference, samples):
    results = []
    for [gt, pred] in tqdm(zip(reference, samples), total=len(reference)):
        results.append(surface_regularity_compare(gt_points=gt, pred_points=pred))

    pred_var_mean = sum([item['pred_surface_variation_mean'] for item in results]) / len(results)
    pred_var_median = sum([item['pred_surface_variation_median'] for item in results]) / len(results)

    gt_var_mean = sum([item['gt_surface_variation_mean'] for item in results]) / len(results)
    gt_var_median = sum([item['gt_surface_variation_median'] for item in results]) / len(results)
    TEMPLATE= 50 * '-' + '\n|' + 16 * ' ' + '{}:{}' + 17 * ' ' + '|\n' + 50 * '-'
    print(TEMPLATE.format('regularity ', f"gt mean={gt_var_mean:.4f} median={gt_var_median:.4f}\n pred mean={pred_var_mean:.4f} median={pred_var_median:.4f}"))
