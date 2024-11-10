# `/weights`

This folder contains weights saved during training of a given model type.

The naming convention for the name corresponds to the network's name and then the number of connections for each layer. The files use PyTorch's `.pth` file extension.

```
<model>_<input>x<output>.pth
```

For example:
```
dqn_8x128x128x4.pth
```