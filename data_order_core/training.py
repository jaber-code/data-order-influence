from torch.amp import autocast, GradScaler
import torch
from utility import *
from torchvision.transforms import v2
from torch.utils.data import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup 
from incremental_scheduler import *
# Using autocast and scaling method from here: https://pytorch.org/docs/stable/notes/amp_examples.html
class Trainer:

    log_file_train = "training_data_log/training_data.csv"
    log_file_test = "training_data_log/testing_data.csv"
    log_file_inc_training = "training_data_log/inc_training_data.csv"

    WARMUP_START_LR = 0.001
    WARMUP_END_LR = 0.1
    MOMENTUM = 0.9
    ETA_MIN = 0.0

    def __init__(self, model, training_params, classes_count, world_size=1, test_loader=None):

        self.model = model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.device = training_params.device
        self.num_epochs = training_params.num_epochs

        self.use_optimizations = training_params.use_optimizations
        self.batch_size = training_params.batch_size
        self.inc_training_type = training_params.inc_training_type
        self.inc_epoch_type = training_params.inc_epoch_type
        self.warmup_epochs = training_params.warmup_epochs

        self.world_size = world_size
        self.classes_count = classes_count
        self.test_loader = test_loader
        
        scaled_lr = (training_params.learning_rate * world_size) * (self.batch_size / 1024)
        self.scaled_lr  = scaled_lr
        print("scaled_lr: ", scaled_lr)

        self.optimizer = optim.SGD(self.model.parameters(), lr=scaled_lr, momentum=0.9)
        cos_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=0.0)

        self.scheduler = create_lr_scheduler_with_warmup( cos_scheduler,
                                                          warmup_start_value=0.001,
                                                          warmup_end_value=0.1,
                                                          warmup_duration=training_params.warmup_epochs)            

    def apply_augmentation(self, inputs, labels, cutmix, mixup):
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        inputs, augmented_labels = cutmix_or_mixup(inputs, labels)
        augmented_labels = torch.argmax(augmented_labels, dim=1)
        return inputs, augmented_labels


    @timer
    def train_with_evaluation(self, train_loader, classes, total_classes, inc=False, epochs=None):
        print("=======================----------- Start Training... ")
        scaler = GradScaler()

        num_of_classes = len(classes)

        mixup = v2.MixUp(num_classes=total_classes)
        cutmix = v2.CutMix(num_classes=total_classes)

        if epochs is None:
            epochs = self.num_epochs

        total_classes_trained = 0
        epochs = int(epochs)
        print("training on ", epochs ," epochs... ")
        self.model.train()
        for epoch in range(epochs): 
            running_loss = 0.0
            print("------------Epoch round ", epoch)
            
            total_samples = 0

            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            correct = 0
            total = 0
            total_classes_trained += num_of_classes
            for batch_index, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                batch_size = inputs.size(0)
                total_samples += batch_size
                augmented_labels = None
                try:
                    inputs, augmented_labels = self.apply_augmentation(inputs, labels, cutmix, mixup)

                    self.optimizer.zero_grad()
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, augmented_labels)

                        if self.world_size > 1:
                            loss_tensor = loss.detach()
                            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                            loss_tensor /= torch.distributed.get_world_size()

                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                        
                        running_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        if batch_index % 200 == 0:
                            print(f"Checking at i={batch_index}, loss={loss}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    print("Inputs shape:", inputs.shape)
                    print("Labels shape before augmentation:", labels.shape)
                    print("Labels data type:", labels.dtype)

            if total_classes_trained >= total_classes:
                self.scheduler(self.optimizer, epoch)
                print("step now at total_classes_trained, ", total_classes_trained , "  total_classes, ", total_classes)
                total_classes_trained = 0

            acc = self.test(self.test_loader, None, classes if inc else None)

            Logger.log_to_csv(Trainer.log_file_train, len(classes), epoch, acc, mode='a')
            print(f" Accuracy After ", epoch, " epochs is ", )

            for param_group in self.optimizer.param_groups:
                print(f"Epoch {epoch}: Current learning rate: {param_group['lr']}")

    @timer
    def test(self, test_loader, acc_list=[], classes_to_include=None):
        print("Testing The model...")
        correct = 0
        total = 0

        if classes_to_include is not None:
            print("test classes_to_include: ")
            test_dataset = test_loader.dataset
            subset_test_dataset = self.get_subset_of_classes(test_dataset, classes_to_include)
            test_loader = torch.utils.data.DataLoader(subset_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            print(" for classes of len ", len(classes_to_include) ) 
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device) 

                if self.use_optimizations:
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        if acc_list is not None:
            acc_list.append(accuracy)
                
        print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        return accuracy

    def start_incremental_training(self, type, train_dataset, train_sampler, all_classes, classes_default_order, initial_num_classes, step_size):

        print("method_to_call: ", type)
        self.incremental_training(train_dataset, train_sampler, all_classes, classes_default_order, initial_num_classes, step_size)

    def incremental_training(self, train_dataset, train_sampler, all_classes, classes_default_order, initial_num_classes, step_size):
        
        print(f"Training on all_classes: {all_classes[:20]}")

        total_classes = len(all_classes)
        #total_samples_normal = self.num_epochs * total_classes
        total_samples_inc = 0

        num_increments = (total_classes - initial_num_classes) // step_size + 1
        init_inc_start_epoch = 0
        end_epochs = 0
        
        scheduler = IncrementalScheduler(step_size, self.num_epochs, total_classes)

        scale_factor = 0               
        if self.inc_epoch_type == "dec":
            init_inc_start_epoch = self.num_epochs * 0.4
            end_epochs = 2
            scale_factor, iters_for_increment = scheduler.estimate_decay(iters_1=init_inc_start_epoch, min_iters=end_epochs)
        elif self.inc_epoch_type == "static":
            scale_factor, iters_for_increment = scheduler.estimate_constant_iters()
        elif self.inc_epoch_type == "inc":
            end_epochs = self.num_epochs * 0.4
            init_inc_start_epoch = 1
            scale_factor, iters_for_increment = scheduler.estimate_scale(iters_1=init_inc_start_epoch, max_iters=end_epochs)

        num_classes_to_train = initial_num_classes
        train_sampler = None
        print("self.inc_epoch_type: ",  self.inc_epoch_type)
        print("num_increments: ",  num_increments)
        print("init_inc_start_epoch: ",  init_inc_start_epoch)
        print("end_epochs: ",  end_epochs)

        class_to_idx = self.get_class_indices_from_all_classes(classes_default_order)
        all_classes_indices = [class_to_idx[cls.split(',')[0]] for cls in all_classes]
        print(f"Training on all_classes_indices: {all_classes_indices[:20]}")

        all_trainloader = self.prepare_subset_trainloader(train_dataset, all_classes_indices, train_sampler)
        total_num_samples = len(all_trainloader)

        increment_iter = 1  
        while num_classes_to_train <= len(all_classes_indices):
            print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

            current_classes = all_classes_indices[:num_classes_to_train]

            print("==epochs for iter ", increment_iter, " ", iters_for_increment)
            print("total_samples_inc: ", total_samples_inc)
            print(f"Training on new classes: {current_classes[len(current_classes)-step_size : len(current_classes)]}")
            
            Logger.log_to_csv_incs_iterations(Trainer.log_file_inc_training, num_classes_to_train, iters_for_increment, mode='a')

            subset_trainloader = self.prepare_subset_trainloader(train_dataset, current_classes, train_sampler)
            num_samples_to_train = len(subset_trainloader)
            print("len num_samples_to_train: ", num_samples_to_train)

            self.update_learning_rate(num_classes_to_train, total_classes)
            self.train_with_evaluation(subset_trainloader, current_classes, total_classes ,True, epochs=round(iters_for_increment))

            iters_for_increment = iters_for_increment * scale_factor
            if self.inc_epoch_type == "dec":
                iters_for_increment = max(iters_for_increment, end_epochs)

            total_samples_inc += (round(iters_for_increment) * num_samples_to_train)

            num_classes_to_train += step_size
            increment_iter += 1
            
        print(f"\nTotal samples in normal training: {total_num_samples}")
        print(f"Total samples in incremental training: {total_samples_inc}")

    def prepare_subset_trainloader(self, train_dataset, current_classes, train_sampler):        
        subset_trainset = self.get_subset_of_classes(train_dataset, current_classes)
        print(f"Subset Trainset Size: {len(subset_trainset)}")

        if torch.cuda.is_available() and torch.distributed.is_initialized():
            train_sampler = DistributedSampler(subset_trainset, shuffle=True)

        return torch.utils.data.DataLoader(subset_trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)

    def update_learning_rate(self, num_trained_classes, total_classes):
        
        adjusted_lr = (self.scaled_lr * num_trained_classes) / total_classes
        self.optimizer = optim.SGD(self.model.parameters(), lr=adjusted_lr, momentum=Trainer.MOMENTUM)

        cos_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=Trainer.ETA_MIN)
        self.scheduler = create_lr_scheduler_with_warmup(cos_scheduler, warmup_start_value=Trainer.WARMUP_START_LR, warmup_end_value=Trainer.WARMUP_END_LR, warmup_duration=self.warmup_epochs)

        return adjusted_lr

    def get_subset_of_classes(self, dataset, classes_to_include):

        subset_indices = [i for i, label in enumerate(dataset.targets) if label in classes_to_include]

        if not subset_indices:
            raise RuntimeError("Filtered dataset is empty! Check class indices.")
        
        subset = torch.utils.data.Subset(dataset, subset_indices)

        return subset


    def get_class_indices_from_all_classes(self, all_classes):
        class_to_idx = {class_name.split(',')[0]: i for i, class_name in enumerate(all_classes)}
        return class_to_idx


