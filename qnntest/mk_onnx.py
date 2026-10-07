import numpy as np
import onnx
from onnx import helper, TensorProto

# decode-like GEMV as MatMul: [1,5120] x [5120,5120] fp16
rng = np.random.default_rng(0)
W = (rng.standard_normal((5120, 5120)) * 0.02).astype(np.float16)
X = (rng.standard_normal((1, 5120))).astype(np.float16)

node = helper.make_node("MatMul", ["x", "w"], ["y"])
gx = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 5120])
gy = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 5120])
gw = helper.make_tensor("w", TensorProto.FLOAT16, [5120, 5120], W.tobytes(), raw=True)
graph = helper.make_graph([node], "gemv", [gx], [gy], [gw])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7
onnx.save(model, r"D:\Download\Bonsai2_NPU\qnntest\gemv.onnx")
np.save(r"D:\Download\Bonsai2_NPU\qnntest\x.npy", X)
print("saved gemv.onnx + x.npy")
