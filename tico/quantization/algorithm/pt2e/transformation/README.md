## transformation

The _transformation_ module provides a set of user-defined transformation passes
 designed to prepare a graph for quantization.

Before annotating the graph, this module allows users to perfrom custom transformation.
 For example, it includes passes that convert scalars in the graph into tensors,
enabling them to be quantized.

### Design Considerations

The modules(passes) in the _transformation_ module are implemented to work with the
 `transformation_for_annotation` interface by the `Quantizer`.

Unlike other passes in `tico.utils.passes`, the passes in this module do **not**
 inherit from the `PassBase` class. This is because the `transformation_for_annotation`
 API in the `Quantizer` uses a `torch.fx.GraphModule` as input, rather than the 
 `ExportedProgram` used by `PassBase`.

```python
# https://github.com/pytorch/pytorch/blob/06b4b96b/torch/ao/quantization/quantizer/quantizer.py#L137-L150
class Quantizer(ABC):
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalares.

        Note: this is an optional method
        """
        return model
```