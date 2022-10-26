# MutableChannels

MutableChannels are used to deal with mutable number of channels in DynamicOps.

```
|-----------------------------------------|
| mutable_in_channel(BaseMutableChannel)  |
| --------------------------------------- |
| DynamicOp                               |
| --------------------------------------- |
| mutable_out_channel(BaseMutableChannel) |
| --------------------------------------- |
```

\`
All MutableChannels inherit from BaseMutableChannel. Each MutableChannel has to implement two property.

- current_choice: get and set the choice of the MutableChannel.
- current_mask: get the channel mask according to the current_choice.

## MutableChannelContainer

Here, we introduce a special MutableChannel: MutableChannelContainer. As the channels of a DynamicOp may belong to different MutableChannelUnits, we use MutableChannelContainers to store multiple MutableChannels as below.

```
-----------------------------------------------------------
|                   MutableChannelContainer               |
-----------------------------------------------------------
|MutableChannel1|     MutableChannel2     |MutableChannel3|
-----------------------------------------------------------
```

MutableChannelContainer has an method to register MutableChannels.

- register_mutable: register/store BaseMutableChannel in the
  MutableChannelContainer
