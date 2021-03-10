# Report for Experiment: q1-depthmap-plaincnn-height-95k-evidential

This report summarizes our experiment, which uses depthmaps as input data
for height prediction. We use a Convolutional Neural Network (CNN).

It focusses on the uncertainty estimation of a model.

## Related work

We will follow the approach presented in the paper: [Deep Evidential Regression](https://arxiv.org/abs/1910.02600).

## Approach

We use the plaincnn network architecture with only one change:
Instead of the last dense layer we use a special layer `edl.layers.DenseNormalGamma(1)`.

We also change the loss function from MSE to `EvidentialRegressionLoss`.
