There are two ways in which SWA is implemented in pytorch.

1. torchcontrib: I personally prefer this as I have used this in past and have good results. 
2. official pytorch.

In the main file,

I implement both these `swa_train_contrib` and `swa_train_pytorch` corresponding to torchcontrib and pytorch respectively.

Under both implementations, training is performed in Dataparallel.

The official pytorch takes average of the model updates after every epoch but I have implemented it to update after every few iterations.
If one wants to use it to update every epoch, just copy the code to update weights after each epoch.

I did a sample run and wrote the output to log.txt -- just start from main.py and see how the code flows.