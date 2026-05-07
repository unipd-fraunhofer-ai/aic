# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from typing import Tuple

import faiss
import faiss.contrib.torch_utils
import torch


def kmeans(
    samples: torch.Tensor,
    num_centroids: int,
    num_iter: int = 50,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K-means clustering.

    Ref:
    [1] https://github.com/facebookresearch/faiss/wiki/FAQ#how-z
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

    Args:
        samples: Samples of shape (num_samples, num_dims).
        num_centroids: Number of centroids/clusters.
        num_iter: Number of k-means iterations.
        verbose: Whether to print progress.
    Returns:
        A tuple of (centroids, cluster_ids, centroid_distances).
    """

    samples = samples.to("cuda").to(torch.float32).contiguous()
    if verbose:
        print(
            f"FAISS k-means on GPU: {samples.shape[0]} samples, "
            f"{samples.shape[1]} dims, {num_centroids} centroids",
            flush=True,
        )

    # Create a k-means object.
    num_dims = samples.shape[1]
    kmeans = faiss.Kmeans(
        num_dims,
        num_centroids,
        niter=num_iter,
        gpu=True,
        verbose=verbose,
        seed=0,
        spherical=False,
    )

    # Cluster the samples.
    kmeans.train(samples.cpu().numpy())

    # Get per-sample cluster assignments.
    centroids = torch.as_tensor(kmeans.centroids, dtype=torch.float32, device=samples.device)
    kmeans.index.reset()
    kmeans.index.add(centroids)
    centroid_distances, cluster_ids = kmeans.index.search(samples, 1)

    # Reshape from (num_samples, 1) to (num_samples).
    centroid_distances = centroid_distances.squeeze(axis=-1)
    cluster_ids = cluster_ids.squeeze(axis=-1)

    return (
        centroids,
        cluster_ids.to(torch.int32).to(samples.device),
        centroid_distances.to(samples.device),
    )
