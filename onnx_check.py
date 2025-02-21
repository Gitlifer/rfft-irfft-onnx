import onnx

def check_onnx_input_output(model_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)
    # 检查模型是否有效
    onnx.checker.check_model(model)

    print("Inputs:")
    # 获取输入信息
    inputs = model.graph.input
    for input_tensor in inputs:
        tensor_name = input_tensor.name
        tensor_type = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
        shape = [dim.dim_value if dim.dim_value != 0 else f"'{dim.dim_param}'" for dim in input_tensor.type.tensor_type.shape.dim]
        shape_str = f"[{', '.join(map(str, shape))}]"
        print(f"  Name: {tensor_name}, Shape: {shape_str}, Type: tensor({tensor_type.lower()})")

    print("\nOutputs:")
    # 获取输出信息
    outputs = model.graph.output
    for output_tensor in outputs:
        tensor_name = output_tensor.name
        tensor_type = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
        shape = [dim.dim_value if dim.dim_value != 0 else f"'{dim.dim_param}'" for dim in output_tensor.type.tensor_type.shape.dim]
        shape_str = f"[{', '.join(map(str, shape))}]"
        print(f"  Name: {tensor_name}, Shape: {shape_str}, Type: tensor({tensor_type.lower()})")


model_path = 'model.onnx'
check_onnx_input_output(model_path)