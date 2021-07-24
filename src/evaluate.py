
import torch

def test(model, sampler, hamiltonian):
    model.eval()
    with torch.no_grad():
        sampler.output_stats = True
        samples, avg_accept = sampler()
        sampler.output_stats = False
        samples = torch.tensor(samples).float()
        scores, _ = hamiltonian.compute_local_energy(samples, model)
        if scores.dtype == torch.complex64:
            scores = scores.real
        score = scores.mean()
        max_score = scores.min()
        variance = torch.var(scores)
    return score, max_score, variance, avg_accept