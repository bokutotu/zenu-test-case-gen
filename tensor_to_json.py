def tensor_to_dict(tensor):
    shape = tensor.shape
    stride = tensor.stride()
    shape = list(shape)
    stride = list(stride)
    flatten_tensor = tensor.flatten()
    data = flatten_tensor.detach().cpu().tolist()
    ptr_offset = 0
    data_type = "f32"
    return {
        "shape": shape,
        "stride": stride,
        "data": data,
        "ptr_offset": ptr_offset,
        "data_type": data_type
    }

