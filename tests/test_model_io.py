import tempfile,unittest
from pathlib import Path
import torch
from model_io import FORMAT,load_inference_model,save_inference_model

class PortableModelTest(unittest.TestCase):
    def test_fp32_round_trip_and_no_optimizer(self):
        torch.manual_seed(42)
        model=torch.nn.Linear(4,3).eval()
        config={'_target_':'torch.nn.Linear','in_features':4,'out_features':3}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            save_inference_model(model,config,path,{'steps':32866})
            payload=torch.load(path,weights_only=True)
            self.assertEqual(payload['format'],FORMAT)
            self.assertNotIn('optimizer',payload)
            loaded=load_inference_model(path)
            x=torch.randn(8,4)
            torch.testing.assert_close(model(x),loaded(x),rtol=0,atol=0)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(p.dtype==torch.float32 for p in loaded.parameters()))

    def test_reject_unknown_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.pt';torch.save({'state_dict':{}},path)
            with self.assertRaises(ValueError):load_inference_model(path)
