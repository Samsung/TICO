import torch
from torch.export import Dim
class SimpleCopyWithBroadcastToDynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(1, 5)), {}
    
    def get_dynamic_shapes(self):
        dim = Dim("dim", min=1, max=128)
        dynamic_shapes = {
            "dst": {0: dim},
            "src": {},
            }
        return dynamic_shapes

model = SimpleCopyWithBroadcastToDynamicShape()    
    
ep = torch.export.export(
    model,
    args=(torch.randn(5, 5), torch.randn(1, 5)),
    dynamic_shapes={"dst": {0: Dim("dim", min=1, max=128)}, "src": {}}
)

breakpoint()
