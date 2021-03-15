# LEAF-TORCH: A Pytorch Benchmark for Federated Learning
LEAF-Torch is a torch implementation of LEAF, which is originally implemented by TensorFlow.

More details about LEAF can be found at:

-   Github:  [https://github.com/TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf)
- LEAF-MXNET (By Dr Li) [https://github.com/Lizonghang/leaf-mx](https://github.com/Lizonghang/leaf-mx)
-   Documentation:  [https://leaf.cmu.edu/](https://leaf.cmu.edu/)
    
-   Paper: "[LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097)"

## Note

- Go to directory of specific dataset to find the instruction of generating data
- Linux environment is required to run this demo(If your training&test dataset is already prepared,feel free to run this demo in windows)
## Environment &Installation
- Python 3.6
- Pytorch  `conda install torch`
- Numpy   `conda install numpy`
- Pandas   `conda install pandas`

## Instruction
Default values for hyper-parameters are set in `utils/args.py`, including:
| Variable Name | Default Value|Optional Values|Description|
|--|--|--|--|
| -dataset|"femnist"|"femnist"|Dataset used for federated learning|
|-model|"cnn"|"cnn"|Neural network used for federated training.|
|--num-rounds|2|integer|Number of rounds to simulate|
|--eval-every|20|integer|Evaluate the federated model every few rounds.
|--clients-per-round|10|integer|Number of clients participating in each round.
|--batch-size|5|integer|Number of training samples in each batch.
|--num-epochs|2|integer|Number of local epochs in each round.|
|-lr|0.01|float|Learning rate for local optimizers.
|--seed|0|integer|Seed for random client sampling and batch splitting.
|--metrics-name|"metrics"|string|Name of matrics files
|--log-dir|"logs"|string|Directory for log files.
|--log-rank|0|integer|Identity for current training process (i.e., `CONTAINER_RANK`). Log files will be written to `logs/{DATASET}/{CONTAINER_RANK}/` (e.g., `logs/femnist/0/`)
|--use-val-set|None|None|Set this option to use validate set otherwise the test set is used
|--count-ops|None|None|Set this option to enable operation counter, otherwise `flops=0` is returned. Enable this will increase the CPU usage and reduce efficiency.
## Othernotes
- You can find the output of this demo in logs folder. e.g`logs/femnist/0`
- More functions will be uploaded soon.
## Thanks

Thanks to [Dr Zonghang Li](https://github.com/Lizonghang)  ,who gives this project  valuable advice and instruction.

And if this project is useful to you,don't forget to give me a star~~~



