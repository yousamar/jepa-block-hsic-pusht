import ast
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf
from ldep import block_dependence_loss, hsic


class DependenceTest(unittest.TestCase):
    def test_nonlinear_dependence(self):
        torch.manual_seed(231)
        x, independent = torch.randn(1024, 1), torch.randn(1024, 1)
        dependent = (x.square() - 1) / 2 ** .5
        values = {'nonlinear': hsic(x, dependent).item(), 'independent': hsic(x, independent).item()}
        print('NONLINEAR_CHECK', values, flush=True)
        self.assertGreater(values['nonlinear'], 5 * values['independent'])

    def test_finite_gradient_all_blocks_and_no_rng(self):
        torch.manual_seed(391)
        emb = torch.randn(64, 4, 192, requires_grad=True)
        before = torch.random.get_rng_state().clone()
        loss = block_dependence_loss(emb)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        for grad in emb.grad.chunk(4, -1):
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_constants_and_time_axis(self):
        # Different times must never be interpreted as independent batch samples.
        emb = torch.arange(4.).view(1, 4, 1).expand(12, 4, 192).clone().requires_grad_()
        loss = block_dependence_loss(emb)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())

    def test_attached_bandwidth_radial_gradient(self):
        torch.manual_seed(192)
        z=torch.randn(32,2,192,requires_grad=True)
        value=block_dependence_loss(z)
        gradient,=torch.autograd.grad(value,z)
        self.assertLess(abs(float((gradient*z).sum())),1e-6)
        torch.testing.assert_close(value,block_dependence_loss(z.detach()*2),atol=1e-7,rtol=1e-5)

    def test_invalid_shape(self):
        for shape, blocks in [((1, 4, 192), 4), ((4, 4, 191), 4), ((4, 4, 192), 1)]:
            with self.assertRaises(ValueError):
                block_dependence_loss(torch.zeros(shape), blocks)

    def test_zero_weight_matches_original_forward_and_gradients(self):
        import train
        from module import SIGReg
        original = (Path(__file__).parent / 'upstream_forward.py').read_text()
        fn = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'lejepa_forward')
        namespace = dict(train.__dict__)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original_snapshot>', 'exec'), namespace)

        class TinyWorld(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(192, 192)
                self.action_encoder = nn.Linear(10, 192)
                self.predictor = nn.Sequential(nn.Linear(384, 192), nn.Dropout(.1))
                self.norm = nn.BatchNorm1d(192)
            def encode(self, batch):
                batch['emb'] = self.encoder(batch['pixels'])
                batch['act_emb'] = self.action_encoder(batch['action'])
                return batch
            def predict(self, x, a):
                y = self.predictor(torch.cat([x, a], -1))
                return self.norm(y.flatten(0, 1)).reshape_as(y)
        class Harness:
            def __init__(self, model):
                self.model = model
                self.sigreg = SIGReg(num_proj=32)
            def log_dict(self, *args, **kwargs):
                pass

        cfg = OmegaConf.create(dict(history_size=3, num_preds=1, aux=dict(enabled=False),
                                   loss=dict(sigreg=dict(weight=.09), dep=dict(weight=0, blocks=4))))
        torch.manual_seed(432)
        original_model = TinyWorld().train()
        new_model = copy.deepcopy(original_model)
        batch = {'pixels': torch.randn(12, 4, 192), 'action': torch.randn(12, 4, 10)}
        torch.manual_seed(777)
        old = namespace['lejepa_forward'](Harness(original_model), copy.deepcopy(batch), 'train', cfg)
        old['loss'].backward()
        old_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(777)
        with patch.object(train, 'block_dependence_loss', side_effect=AssertionError('zero weight must skip HSIC')):
            new = train.lejepa_forward(Harness(new_model), copy.deepcopy(batch), 'train', cfg)
        new['loss'].backward()
        self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))
        self.assertEqual(set(old), set(new))
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in original_model.parameters()), sum(p.numel() for p in new_model.parameters()))
        for (name, p), (_, q) in zip(original_model.named_parameters(), new_model.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
        for key, value in original_model.state_dict().items():
            torch.testing.assert_close(value, new_model.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
