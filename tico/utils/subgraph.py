import torch
from torch.export import ExportedProgram
from typing import Optional
import functools

_gm_map = None
def get_gm_map(ep: Optional[ExportedProgram] = None):
    """
    Returns [{"index":0, "name": "true_graph_0", "getter": lambda ep: ep.graph_module}, ...}]
    """
    # Build _gm_map only once while compiler running
    global _gm_map
    if _gm_map is None:
        assert ep is not None
        _gm_map = _build_gm_map(ep)
    return _gm_map

def _build_gm_map(ep: ExportedProgram):
    ret = []
    
    # root GraphModule 추가
    ret.append({
        "index": len(ret),
        "name": "",
        "gm": ep.graph_module,
    })
    
    # Inspect non-root subgraphs
    for node in ep.graph.nodes:
        if node.op == "get_attr":
            attr = getattr(node.graph.owning_module, node.target)
            
            # TODO: Enable recursion (n-depth)
            if isinstance(attr, torch.fx.graph_module.GraphModule):
                assert hasattr(node, 'name')
                assert getattr(node, 'name') != ret[0]["name"]
                graph_name = getattr(node, 'name')
                ret.append({
                    "index": len(ret),
                    "name": graph_name,
                    "gm": attr,
                })
    return ret
