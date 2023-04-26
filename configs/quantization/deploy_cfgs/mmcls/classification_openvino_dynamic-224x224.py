deploy_cfg = dict(
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['input'],
        output_names=['output'],
        input_shape=None,
        optimize=True,
        dynamic_axes={
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output': {
                0: 'batch'
            }
        }),
    backend_config=dict(
        type='openvino',
        model_inputs=[dict(opt_shapes=dict(input=[1, 3, 224, 224]))]),
    codebase_config=dict(type='mmcls', task='Classification'),
    function_record_to_pop=[
        'mmcls.models.classifiers.ImageClassifier.forward',
        'mmcls.models.classifiers.BaseClassifier.forward'
    ],
)
