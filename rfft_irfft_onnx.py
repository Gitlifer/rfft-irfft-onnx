import torch
import torch.onnx
import onnxruntime as ort

# 和torch.fft.rfft运算不相同，只相当于rfft_matrix
def custom_rfft(g, X, n, dim, norm):
    x = g.op(
        "Unsqueeze",
        X,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
    )
    x = g.op(
        "Unsqueeze",
        x,
        g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
    )
    x = g.op("DFT", x, axis_i=1, inverse_i=0, onesided_i=1)
    x = g.op(
        "Squeeze",
        x,
        g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
    )
    return x

# 注册自定义操作符
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::fft_rfft",
    symbolic_fn=custom_rfft,
    opset_version=OPSET_VERSION,
)

# 定义一个简单的 PyTorch 模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = self.fft_size = 10
        
        self.register_buffer(
            "rfft_matrix",
            torch.view_as_real(torch.fft.rfft(torch.eye(self.window_size))).transpose(
                0, 1
            ),
        )
        self.register_buffer("irfft_matrix", torch.linalg.pinv(self.rfft_matrix))
    def forward(self, x):
        spec = torch.matmul(x, self.rfft_matrix).transpose(0,1)
        wav_out = (
            torch.einsum("bfi,fij->bj", spec, self.irfft_matrix)
        )
        return spec, wav_out

model = MyModel()
input_tensor = torch.randn(1,10) # [channel, length]

# 导出模型为 ONNX 格式
torch.onnx.export(
    model,
    input_tensor,
    "model.onnx",
    input_names=["inp"],
    opset_version=17,
)

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'inp':input_tensor.numpy()})

# check
print(outputs[0])
print(torch.fft.rfft(input_tensor))

print(input_tensor)
print(outputs[1])