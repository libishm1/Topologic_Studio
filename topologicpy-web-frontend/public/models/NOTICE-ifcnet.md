# IFCNet geometry classifier

`ifcnet-dgcnn-2048.onnx` is IFCNet's pretrained DGCNN point-cloud classifier, converted to ONNX, with its edge convolutions rewritten exactly and the channels training left unused pruned. Its answers are the same as the original network's. It was produced by `tools/identify-ifcnet.py` in this repository.

- Paper: Emunds, C., Pauen, N., Richter, V., Frisch, J. and van Treeck, C. (2021). IFCNet: A Benchmark Dataset for IFC Entity Classification. EG-ICE 2021. arXiv:2106.09712, https://doi.org/10.48550/arXiv.2106.09712
- Code: https://github.com/RWTH-E3D/ifcnet-models, MIT licence, Copyright (c) 2021 Christoph Emunds.
- Weights: `dgcnn_model` from https://ifcnet.e3d.rwth-aachen.de/, published by the authors. No licence is stated for the weights or for the dataset they were trained on. They are included here with this credit. They will be removed if the authors ask; contact the repository owner.

SHA-256 of `ifcnet-dgcnn-2048.onnx`: `6f1fc45ceaabb7ce1538e6dce3caef09990026da792aee7d3358018f6ef85481`
