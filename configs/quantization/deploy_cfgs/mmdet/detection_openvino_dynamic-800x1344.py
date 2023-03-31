deploy_cfg = dict(
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_shape=None,
        input_names=['input'],
        output_names=['dets', 'labels'],
        optimize=True,
        dynamic_axes={
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }),
    backend_config=dict(
        type='openvino',
        model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))]),
    codebase_config=dict(
        type='mmdet',
        task='ObjectDetection',
        model_type='end2end',
        post_processing=dict(
            score_threshold=0.05,
            confidence_threshold=0.005,  # for YOLOv3
            iou_threshold=0.5,
            max_output_boxes_per_class=200,
            pre_top_k=5000,
            keep_top_k=100,
            background_label_id=-1,
        )),
    function_record_to_pop=[
        'mmdet.models.detectors.single_stage.SingleStageDetector.forward',
        'mmdet.models.detectors.two_stage.TwoStageDetector.forward',
        'mmdet.models.detectors.single_stage_instance_seg.SingleStageInstanceSegmentor.forward'  # noqa: E501
    ])
