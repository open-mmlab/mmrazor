# How to make your model prunable by MMRazor

We make much effort to make MMRazor can prune models automatically. However, there are also some cases we need help to handle. In this tutorial, we will show you how to make your model prunable by MMRazor when MMRazor can't take it automatically.

We introduce you to how MMRazor prunes a model and what requirements are needed for the models in each step.

1. First, we need a demo input for your model to conduct forward. We have implemented some demo inputs for some models. You can implement your demo input if they can't satisfy your model. Please refer to [demo input](../../../../mmrazor/models/task_modules/demo_inputs/demo_inputs.py) for more details.
2. We will trace your model using Fx tracer. Fx tracer requires models to satisfy some requirements to make sure the models are traceable. MMRazor has a modified fx tracer to make it more robust by automatically wrapping untraceable parts. However, it's also limited. It would help to change your model to make it traceable when tracing fails. Please refer to [fx tracer](https://pytorch.org/docs/stable/fx.html) for more details.
3. Then, we will convert the traced graph to a ChannelGraph. ChannelGraph is used to analyze the channel dependency with ChannelNodes. Please implement ChannelNodes for the ops that are not pre-defined in MMRazor. Please refer to [ChannelNode](../../../../mmrazor/structures/graph/channel_nodes.py) for more details.

The above steps should be able to solve most of problems, if not, please feel free to open an issue.
