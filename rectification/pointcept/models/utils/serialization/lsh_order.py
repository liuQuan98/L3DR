"""
Author: Quan Liu
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn.functional as F


# Modified from 'Reformer: The Efficient Transformer'
# https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py
def hash_vectors(n_buckets: int,
                 vecs: torch.Tensor,
                 n_hashes=8,
                 dropout=0.,
                 ):
    batch_size = vecs.shape[0]
    device = vecs.device

    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    assert n_buckets % 2 == 0

    rot_size = n_buckets

    # [b,f,h,i]
    rotations_shape = (
        1,
        vecs.shape[-1],
        n_hashes,
        rot_size // 2)
    
    # [b,t,h,f]
    translations_shape = (
        1,
        1,
        n_hashes,
        vecs.shape[-1])
    
    scale = torch.tensor([40, 40, 10], device=device).reshape((1,1,1,3)) # random bias for the rotation center

    random_rotations = torch.randn(rotations_shape, device=device).expand(batch_size, -1, -1, -1)
    random_translations = torch.randn(translations_shape, device=device) * scale

    # original implementation: only rotation from origin.
    # dropped_vecs = F.dropout(vecs, p=dropout)
    # rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

    # our implementation: with both random translation and rotation for different hashes.
    dropped_vecs = F.dropout(vecs, p=dropout)
    dupped_vecs = torch.unsqueeze(dropped_vecs, dim=2).expand(-1, -1, n_hashes, -1)
    rotated_vecs = torch.einsum('bthf,bfhi->bhti', (dupped_vecs + random_translations), random_rotations)

    # rotated_vectors size [batch,n_hash,seq_len,buckets]
    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
    buckets = torch.argmax(rotated_vecs, dim=-1)

    # Todo: this is a naive implementation. Improvements needed.
    # buckets is now [batch, n_hashes, seq_len].
    # We treat different hashes as the coordinates of base power n_buckets.
    # Therefore all hashes are converted back to a number (although they do have different weights, which might be a problem).
    offsets = torch.arange(n_hashes, device=device)
    offsets = offsets/(n_hashes/8) if (n_hashes>8) else offsets
    offsets = torch.pow(torch.ones_like(offsets)*n_buckets, offsets).unsqueeze(0).unsqueeze(2)
    buckets = torch.sum(buckets * offsets, dim=1).long()

    # [batch, seqlen]
    return buckets


def lsh_encode(grid_coord: torch.Tensor):
    # we block the support to batch, maintain batched code in Point class
    code = hash_vectors(n_buckets=20, vecs=grid_coord.unsqueeze(0), n_hashes=200, dropout=0.)
    return code[0]
