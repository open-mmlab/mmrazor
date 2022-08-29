# Policy for CIFAR, refer to
# https://github.com/DeepVoltaire/AutoAugment/blame/master/autoaugment.py
policy_cifar = [
    # Group 1
    [
        dict(type='Invert', prob=0.1),
        dict(type='Contrast', magnitude=0.5, prob=0.2)
    ],
    [
        dict(type='Rotate', angle=10., prob=0.7),
        dict(type='Translate', magnitude=150 / 331, prob=0.3)
    ],
    [
        dict(type='Sharpness', magnitude=0.9, prob=0.8),
        dict(type='Sharpness', magnitude=0.3, prob=0.9)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 8,
            direction='vertical',
            prob=0.5),
        dict(
            type='Translate',
            magnitude=150 / 331,
            direction='vertical',
            prob=0.3)
    ],
    [dict(type='AutoContrast', prob=0.5),
     dict(type='Equalize', prob=0.9)],
    # Group 2
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 7,
            direction='vertical',
            prob=0.2),
        dict(type='Posterize', bits=5, prob=0.3)
    ],
    [
        dict(type='ColorTransform', magnitude=0.3, prob=0.4),
        dict(type='Brightness', magnitude=0.7, prob=0.7)
    ],
    [
        dict(type='Sharpness', magnitude=1.0, prob=0.3),
        dict(type='Brightness', magnitude=1.0, prob=0.7)
    ],
    [dict(type='Equalize', prob=0.6),
     dict(type='Equalize', prob=0.5)],
    [
        dict(type='Contrast', magnitude=0.6, prob=0.6),
        dict(type='Sharpness', magnitude=0.4, prob=0.8),
    ],
    # Group 3
    [
        dict(type='ColorTransform', magnitude=0.6, prob=0.7),
        dict(type='Translate', magnitude=150 / 331 / 9 * 7, prob=0.5)
    ],
    [dict(type='Equalize', prob=0.3),
     dict(type='AutoContrast', prob=0.4)],
    [
        dict(
            type='Translate',
            magnitude=150 / 331 / 9 * 2,
            direction='vertical',
            prob=0.4),
        dict(type='Sharpness', magnitude=0.5, prob=0.2)
    ],
    [
        dict(type='Brightness', magnitude=0.5, prob=0.9),
        dict(type='ColorTransform', magnitude=0.7, prob=0.2),
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 7, prob=0.5),
        dict(type='Invert', prob=0.0),
    ],
    # Group 4
    [dict(type='Equalize', prob=0.2),
     dict(type='AutoContrast', prob=0.6)],
    [dict(type='Equalize', prob=0.2),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='ColorTransform', magnitude=0.9, prob=0.9),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='AutoContrast', prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 1, prob=0.2),
    ],
    [
        dict(type='Brightness', magnitude=0.3, prob=0.1),
        dict(type='ColorTransform', magnitude=0.0, prob=0.7)
    ],
    # Group 5
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.4),
        dict(type='AutoContrast', prob=0.9)
    ],
    [
        dict(
            type='Translate',
            magnitude=150 / 331,
            direction='vertical',
            prob=0.9),
        dict(
            type='Translate',
            magnitude=150 / 331,
            direction='vertical',
            prob=0.7)
    ],
    [
        dict(type='AutoContrast', prob=0.9),
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.8)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Invert', prob=0.1)],
    [
        dict(
            type='Translate',
            magnitude=150 / 331,
            direction='vertical',
            prob=0.7),
        dict(type='AutoContrast', prob=0.9)
    ]
]
