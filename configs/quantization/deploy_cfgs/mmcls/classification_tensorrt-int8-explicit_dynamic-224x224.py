deploy_cfg = dict(
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['input'],
        output_names=['output'],
        input_shape=[224, 224],
        optimize=True,
        dynamic_axes=dict(
            input=dict({
                0: 'batch',
                2: 'height',
                3: 'width'
            }),
            output=dict({0: 'batch'}))),
    codebase_config=dict(type='mmcls', task='Classification'),
    backend_config=dict(
        type='tensorrt',
        common_config=dict(
            fp16_mode=False,
            max_workspace_size=1073741824,
            int8_mode=True,
            explicit_quant_mode=True),
        model_inputs=[
            dict(
                input_shapes=dict(
                    input=dict(
                        min_shape=[1, 3, 224, 224],
                        opt_shape=[4, 3, 224, 224],
                        max_shape=[8, 3, 224, 224])))
        ]),
    function_record_to_pop=[
        'mmcls.models.classifiers.ImageClassifier.forward',
        'mmcls.models.classifiers.BaseClassifier.forward', 'torch.cat'
    ],
)
