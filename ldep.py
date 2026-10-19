"""Training-only block-versus-rest nonlinear dependence loss; no parameters."""
import torch


def rbf_gram(x):
    """Median-bandwidth RBF, differentiating its bandwidth in fp32."""
    x = x.float()
    distances = (x[:, None, :] - x[None, :, :]).square().sum(-1)
    off_diagonal = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    positive = distances[off_diagonal & (distances.detach() > 0)]
    bandwidth_squared = (positive.median() if positive.numel() else
                         distances.new_tensor(1e-6)).clamp_min(1e-6)
    return torch.exp(-distances / (2 * bandwidth_squared))


def center_gram(gram):
    return gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()


def hsic(x, y):
    """Biased nonnegative HSIC estimator; observations are rows."""
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError('HSIC requires matching two-dimensional inputs with B >= 2')
    with torch.autocast(device_type=x.device.type, enabled=False):
        gx, gy = center_gram(rbf_gram(x)), center_gram(rbf_gram(y))
        return ((gx * gy).sum() / (len(x) - 1) ** 2).clamp_min(0)


def block_dependence_loss(emb, blocks=4):
    """Mean HSIC(block, remaining blocks), independently at each time index."""
    if emb.ndim != 3 or emb.shape[0] < 2 or emb.shape[1] < 1:
        raise ValueError('Expected embeddings [B >= 2, T >= 1, D]')
    if blocks < 2 or emb.shape[-1] % blocks:
        raise ValueError('D must be divisible by blocks >= 2')
    with torch.autocast(device_type=emb.device.type, enabled=False):
        parts = emb.float().chunk(blocks, dim=-1)
        return torch.stack([
            hsic(parts[k][:, t], torch.cat([parts[j][:, t] for j in range(blocks) if j != k], -1))
            for t in range(emb.shape[1]) for k in range(blocks)
        ]).mean()
