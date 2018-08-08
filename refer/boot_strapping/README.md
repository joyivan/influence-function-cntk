codes

- cross_validate_emnist_resnet.py
data analysis
check how much data is necessary for the neural classifier

- noisy_annotation.py
make noisy annotation
(file path should be changed)

- noisy_emnist_resnet.py
train neural networks among various noise ratios
check how big performance degradation will occur w.r.t. noise ratio

- recovery_annotation.py
after training a neural classifier, this code will recover noisily labeled data by exploiting classifier information (e.g. loss, prediction entropy, etc.)

- ???.py
after recovering the noised dataset, train a new neural network w.r.t. recovered datset

