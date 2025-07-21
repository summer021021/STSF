import os
import ray
import torch
import tempfile

from utils import *
from block import *
from dataset import *

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ray import tune, train
from ray.tune.schedulers import FIFOScheduler


def trainable(config):
    set_random_seed(config["seed"])

    epochs = config["epochs"]

    trainloader,testloader = MNISTLoader(config["batch_size"],config["T"])
    network = Net([config["inSize"]]+ [config["layerSize"]] * config["nlayers"],nclass=config["nclass"],
                  threshold=config["threshold"], decay=config["decay"], vmax=config["vmax"])
    optimizer = Adam(network.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=epochs // 4, gamma=0.5)


    epochs = config["epochs"]
    for epoch in range(epochs):
        training_loss, training_accuracy = fit(network, trainloader, optimizer)
        testing_accuracy = test(network, testloader)
        scheduler.step()
        metrics = {"testing_accuracy": testing_accuracy,
                   "training_accuracy": training_accuracy,
                   "training_loss":training_loss}
        train.report(metrics=metrics)

def main():
    search_space={
        "epochs":100,
        "batch_size":128,

        "lr":5e-4,
        "T":10,
        "threshold":1.0,
        "decay":1.0,
        "vmax":1.0,

        "nclass":10,
        "inSize":784,
        "nlayers":2,
        "layerSize":800,

        "seed":0,
    }
    scheduler = FIFOScheduler()

    storage_path = os.path.abspath("../Log")
    
    exp_name = "STSF_MNIST"
    path = os.path.join(storage_path, exp_name)


    trainable_with_resources = tune.with_resources(
        trainable=trainable,
        resources=train.ScalingConfig(
            trainer_resources={"CPU":16, "GPU":1},
            use_gpu=True,
            placement_strategy="SPREAD",
        )
    )

    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path, 
            trainable_with_resources, 
            param_space=search_space,
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=1,
                scheduler=scheduler,
            ),
            
            run_config=train.RunConfig(
                local_dir=storage_path,
                name=exp_name
            ),
        )

    tuner.fit()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ray.init(num_cpus=16, num_gpus=1,)
    main()
