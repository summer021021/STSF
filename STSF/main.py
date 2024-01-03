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
from ray.tune.schedulers import HyperBandForBOHB, FIFOScheduler
from ray.tune.search.bohb import TuneBOHB


def trainable(config):
    set_random_seed(config["seed"])

    epochs = config["epochs"]

    trainloader,testloader = NMNISTLoader(config["batch_size"],config["T"])
    network = Net([config["inSize"]]+ [config["layerSize"]] * config["nlayers"],nclass=config["nclass"],
                  threshold=config["threshold"], decay=config["decay"], vmax=config["vmax"])
    optimizer = Adam(network.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=epochs // 4, gamma=0.5)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start = checkpoint_dict["epoch"] + 1
            network.load_state_dict(checkpoint_dict["model_state"])

    epochs = config["epochs"]
    for epoch in range(epochs):
        training_loss, training_accuracy = fit(network, trainloader, optimizer)
        testing_accuracy = test(network, testloader)
        scheduler.step()

        metrics = {"testing_accuracy": testing_accuracy,
                   "training_accuracy": training_accuracy,
                   "training_loss":training_loss}
        
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({"epoch": epoch, "model_state": network.state_dict()},os.path.join(tempdir, "checkpoint.pt"))
            train.report(metrics=metrics, checkpoint= train.Checkpoint.from_directory(tempdir))

def main():
    search_space={
        "epochs":100,
        "batch_size":128,

        "lr":5e-5,
        "T":10,
        "threshold":0.3,
        "decay":0.3,
        "vmax":1.0,

        "nclass":10,
        "inSize":784,
        "nlayers":2,
        "layerSize":800,

        "seed":tune.randint(0,999),
    }

    # scheduler = HyperBandForBOHB(
    # time_attr="training_iteration",
    # metric="training_accuracy",
    # mode="max",
    # max_t=100
    # )

    # algo = TuneBOHB(metric="training_accuracy", mode="max")

    scheduler = FIFOScheduler()

    storage_path = "/home/peace/code/Log"
    
    exp_name = "STSF_NMNIST_NH_2_Layers"
    path = os.path.join(storage_path, exp_name)


    trainable_with_resources = tune.with_resources(
        trainable=trainable,
        resources=train.ScalingConfig(
            trainer_resources={"CPU":2, "GPU":0.25},
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
                num_samples=8,
                scheduler=scheduler,
                # search_alg=algo,
            ),
            
            run_config=train.RunConfig(
                local_dir=storage_path,
                name=exp_name
            ),
        )

    tuner.fit()
    
    print("finish!!!")

if __name__ == "__main__":
    # 837013
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    ray.init(num_cpus=16, num_gpus=2,)
    main()
