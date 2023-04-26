deploy_cfg = dict(
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['input'],
        output_names=['dets', 'labels'],
        input_shape=None,
        optimize=True,
        dynamic_axes=dict(
            input=dict({
                0: 'batch',
                2: 'height',
                3: 'width'
            }),
            dets=dict({
                0: 'batch',
                1: 'num_dets'
            }),
            labels=dict({
                0: 'batch',
                1: 'num_dets'
            }))),
    codebase_config=dict(
        type='mmdet',
        task='ObjectDetection',
        model_type='end2end',
        post_processing=dict(
            score_threshold=0.05,
            confidence_threshold=0.005,
            iou_threshold=0.5,
            max_output_boxes_per_class=200,
            pre_top_k=5000,
            keep_top_k=100,
            background_label_id=-1)),
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
                        min_shape=[1, 3, 320, 320],
                        opt_shape=[1, 3, 800, 1344],
                        max_shape=[1, 3, 1344, 1344])))
        ]),
    function_record_to_pop=[
        'mmdet.models.detectors.single_stage.SingleStageDetector.forward',
        'mmdet.models.detectors.two_stage.TwoStageDetector.forward',
        'mmdet.models.detectors.single_stage_instance_seg.SingleStageInstanceSegmentor.forward',  # noqa: E501
        'torch.cat'
    ])
