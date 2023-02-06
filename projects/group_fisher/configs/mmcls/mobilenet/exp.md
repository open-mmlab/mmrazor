| name        | flop  | param | finetune |
| ----------- | ----- | ----- | -------- |
| baseline    | 0.319 | 3.5   | 71.86    |
| fisher_act  | 0.20  | 3.14  | 70.79    |
| fisher_flop | 0.20  | 2.78  | 70.87    |

fisher_act
{
"backbone.conv1.conv\_(0, 32)_32": 21,
"backbone.layer1.0.conv.1.conv_(0, 16)_16": 10,
"backbone.layer2.0.conv.0.conv_(0, 96)_96": 45,
"backbone.layer2.0.conv.2.conv_(0, 24)_24": 24,
"backbone.layer2.1.conv.0.conv_(0, 144)_144": 73,
"backbone.layer3.0.conv.0.conv_(0, 144)_144": 85,
"backbone.layer3.0.conv.2.conv_(0, 32)_32": 32,
"backbone.layer3.1.conv.0.conv_(0, 192)_192": 95,
"backbone.layer3.2.conv.0.conv_(0, 192)_192": 76,
"backbone.layer4.0.conv.0.conv_(0, 192)_192": 160,
"backbone.layer4.0.conv.2.conv_(0, 64)_64": 64,
"backbone.layer4.1.conv.0.conv_(0, 384)_384": 204,
"backbone.layer4.2.conv.0.conv_(0, 384)_384": 200,
"backbone.layer4.3.conv.0.conv_(0, 384)_384": 217,
"backbone.layer5.0.conv.0.conv_(0, 384)_384": 344,
"backbone.layer5.0.conv.2.conv_(0, 96)_96": 96,
"backbone.layer5.1.conv.0.conv_(0, 576)_576": 348,
"backbone.layer5.2.conv.0.conv_(0, 576)_576": 338,
"backbone.layer6.0.conv.0.conv_(0, 576)_576": 543,
"backbone.layer6.0.conv.2.conv_(0, 160)_160": 160,
"backbone.layer6.1.conv.0.conv_(0, 960)_960": 810,
"backbone.layer6.2.conv.0.conv_(0, 960)_960": 803,
"backbone.layer7.0.conv.0.conv_(0, 960)_960": 944,
"backbone.layer7.0.conv.2.conv_(0, 320)\_320": 320
}
fisher_flop
{
"backbone.conv1.conv\_(0, 32)_32": 27,
"backbone.layer1.0.conv.1.conv_(0, 16)_16": 16,
"backbone.layer2.0.conv.0.conv_(0, 96)_96": 77,
"backbone.layer2.0.conv.2.conv_(0, 24)_24": 24,
"backbone.layer2.1.conv.0.conv_(0, 144)_144": 85,
"backbone.layer3.0.conv.0.conv_(0, 144)_144": 115,
"backbone.layer3.0.conv.2.conv_(0, 32)_32": 32,
"backbone.layer3.1.conv.0.conv_(0, 192)_192": 102,
"backbone.layer3.2.conv.0.conv_(0, 192)_192": 95,
"backbone.layer4.0.conv.0.conv_(0, 192)_192": 181,
"backbone.layer4.0.conv.2.conv_(0, 64)_64": 64,
"backbone.layer4.1.conv.0.conv_(0, 384)_384": 169,
"backbone.layer4.2.conv.0.conv_(0, 384)_384": 176,
"backbone.layer4.3.conv.0.conv_(0, 384)_384": 180,
"backbone.layer5.0.conv.0.conv_(0, 384)_384": 308,
"backbone.layer5.0.conv.2.conv_(0, 96)_96": 96,
"backbone.layer5.1.conv.0.conv_(0, 576)_576": 223,
"backbone.layer5.2.conv.0.conv_(0, 576)_576": 241,
"backbone.layer6.0.conv.0.conv_(0, 576)_576": 511,
"backbone.layer6.0.conv.2.conv_(0, 160)_160": 160,
"backbone.layer6.1.conv.0.conv_(0, 960)_960": 467,
"backbone.layer6.2.conv.0.conv_(0, 960)_960": 510,
"backbone.layer7.0.conv.0.conv_(0, 960)_960": 771,
"backbone.layer7.0.conv.2.conv_(0, 320)\_320": 320
}
