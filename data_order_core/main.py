import torch

from dataset.dataset_handler import DatasetHandler
from dataset.dataset_type import DataSetType

from model.model_type import ModelType
from model.model_factory import *
from config.config_manager import ConfigManager
from torch.utils.data import DistributedSampler

import torch._dynamo
from  utility import *
from training import Trainer
import os
import torch.distributed as dist
from dataclasses import dataclass
from enums import *

###
# All Distributed training code were taken from the officla pytorch docs:
# https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
###

torch._dynamo.config.suppress_errors = True


@dataclass
class TrainingParameters:
    device: str                    
    num_epochs: int                
    use_optimizations: bool        
    batch_size: int                  
    warmup_epochs: int     
    learning_rate: int    
    inc_training_type: str
    inc_epoch_type: str 


def retrieve_classes( order):
    file_names = {
        Incremental_Type.ALPH: "generated_files/alphabetical_order.txt",
        Incremental_Type.RAND: "generated_files/random_order.txt",
        Incremental_Type.MOST_DISS_1ST: "generated_files/most_dissimilar_first.txt",
        Incremental_Type.MOST_SIM_1ST: "generated_files/most_similar_first.txt",
        Incremental_Type.HYBRID_DISS_SIM: "generated_files/hybrid-dis-sim.txt",
        Incremental_Type.HYBRID_DISS_SEQ: "generated_files/hybrid-dis-seq.txt"
    }

    file_name = file_names[order]
    print(f"Reading from {file_name}...")

    classes = []
    with open(file_name, 'r') as file:
        for line in file:
            classes.append(line.strip())

    return classes
    
@timer
def __main__():
    print("Starting..")

########### Config ###########   
    config = ConfigManager("data_order_core/config/config.yaml" if os.path.exists("./data_order_core") else "config/config.yaml")

    selected_model = ModelType(config.get_selected_model())
    select_dataset_type = DataSetType(config.get_select_dataset_type())
    epochs = config.get_epochs()
    is_local = config.get_is_local()
    use_optimizations = config.get_use_optimizations()
    imagenet_path = config.get_imagenet_path() 
    initial_num_classes = config.get_initial_num_classes()
    step_num_classes = config.get_step_num_classes()
    batch_size = config.get_batch_size()
    perform_inc_training = config.get_perform_inc_training()
    perform_normal_training = config.get_perform_normal_training()
    inc_training_type = config.get_inc_training_type()
    learning_rate = config.get_learning_rate()
    inc_epoch_type = config.get_inc_decay_type()

    print("======= Used Configs: ")
    print(selected_model.value)
    print(select_dataset_type.value)
    print("epochs: ", epochs)
    print("use_optimizations: ", use_optimizations)
    print("imagenet_path: ", imagenet_path)
    print("initial_num_classes: ", initial_num_classes, " step_num_classes: ", step_num_classes)
    print("batch_size: ", batch_size)
    print("perform_normal_training: ", perform_normal_training)
    print("perform_inc_training: ", perform_inc_training)
    print("learning_rate: ", learning_rate)
    print("inc_epoch_type: ", inc_epoch_type)
    print("inc_training_type: ", inc_training_type)

    print("======= ")
    print("What's new: " + " inc with new dynamic lr (based on scaled) + step based on classes numbers - MIXUP OFF")

    world_size = 1 #  inc with new dynamic lr/wu (remove hc 0.2) + step based on classes numbers - MIXUP OFF
    local_rank = 0

    if torch.cuda.is_available():    
        torch.multiprocessing.set_start_method('forkserver', force=True)
        world_size = int(os.environ['WORLD_SIZE'])
    
    assert world_size > 0, "Invalid world_size; must be greater than 0."
    print("world_size: ", world_size)

    if torch.cuda.is_available():
        local_rank = int(os.environ['LOCAL_RANK']) 
        print("local_rank: ", local_rank)
        dist.init_process_group(backend='nccl') 
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')  
        print("Distributed initialized with NCCL backend.")
    else:
        print("Distributed training is not available;")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########### Dataset Loading ###########
    dataset_handler = DatasetHandler(dataset_type=select_dataset_type, dataset_path=imagenet_path, is_local=is_local, batch_size=batch_size)
    train_dataset, trainloader, testloader, classes, train_sampler = dataset_handler.load_data_set()

    ########### Model init & (compile) ###########
    net = ModelFactory.create_model(selected_model, len(classes))
    net = net.to(device)

    if use_optimizations:
        print("=- Use Compile")
        net = torch.compile(net, mode='max-autotune')
    
    if torch.cuda.is_available() and dist.is_available():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    ########### Params Init For Training ###########
    training_params = TrainingParameters(
        device=device,
        num_epochs=epochs,
        use_optimizations=use_optimizations,
        batch_size=batch_size,
        warmup_epochs=5,
        learning_rate=learning_rate,
        inc_training_type=inc_training_type,
        inc_epoch_type=inc_epoch_type
    )

    if perform_normal_training:        
        print("============ Starts normal training on all data\n")
        trainer = Trainer(net, training_params, world_size=world_size, classes_count=len(classes), test_loader=testloader)   

        trainer.train_with_evaluation(trainloader, classes, len(classes) ,False)

    if perform_inc_training:
        print("============ Starts incremental training\n")
        trainer = Trainer(net, training_params, world_size=world_size, classes_count=len(classes), test_loader=testloader)   

        inc_type = Incremental_Type(inc_training_type)
        print("inc_type: ", inc_type)
        ordered_classes = retrieve_classes(inc_type)

        print("=-= classes: ", classes[:10])
        print("=-= ordered_classes: ", ordered_classes[:10])

        trainer.start_incremental_training(inc_type, train_dataset, train_sampler, all_classes=ordered_classes, classes_default_order=classes , initial_num_classes=initial_num_classes,step_size= step_num_classes)

    if dist: 
        dist.destroy_process_group()
    print(" ================== Finished All")

if __name__ == '__main__':
    try:
        __main__()
    except Exception as e:
        print(f"Error: {e}")
        dist.destroy_process_group()