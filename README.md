# skeleton-LSTMs
An implementation of the paper "Skeleton-based gait index estimation with LSTMs" (IEEE ICIS, Singapore, June 2018)

## Requirements
* Python
* Numpy
* TensorFlow
* Scikit-learn

## Notice
* The code was implemented to directly work on [DIRO gait dataset](http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/)
* Please download the [skeleton data](http://www.iro.umontreal.ca/~labimage/GaitDataset/skeletons.zip) and put the npz file into the folder **dataset**

## Usage
```
python3 LSTM_AE_gait.py
```
The LSTM_AE class was modified from [iwyoo's work](https://github.com/iwyoo/LSTM-autoencoder)

## Example of output
Default training and test sets
```
Finish loading data
(1000, 12, 17)
(400, 12, 17)
(3200, 12, 17)
Tensor("Placeholder:0", shape=(50, 12, 17), dtype=float32)
Training axis X with 100 epochs...
(50, 12, 17)
(50, 12, 17)
X-axis
AUC = 0.834
Training axis Y with 100 epochs...
(50, 12, 17)
(50, 12, 17)
Y-axis
AUC = 0.822
Training axis Z with 100 epochs...
(50, 12, 17)
(50, 12, 17)
Z-axis
AUC = 0.742
=== summation ===
per-segment (non-weighted sum)
AUC = 0.860
per-sequence (non-weighted sum)
AUC = 0.922
per-segment (weighted sum)
AUC = 0.904
per-sequence (weighted sum)
AUC = 0.953

```
