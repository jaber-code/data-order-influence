import time
import builtins
from datetime import datetime
import csv
import os
import torch

class Logger:
    log_file_train = "training_data_log/training_data.csv"
    log_file_inc_training = "training_data_log/inc_training_data.csv"
    original_print = builtins.print

    @staticmethod
    def print_with_timestamp(*args, **kwargs):
        local_rank = 0
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if local_rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            Logger.original_print(f"{timestamp}", *args, **kwargs, flush=True)

    
    @staticmethod
    def get_timestamped_filename(file_path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)
        return os.path.join(dir_name, f"{name}_{timestamp}{ext}")
    
    file_name_normal = get_timestamped_filename(log_file_train)
    file_name_inc = get_timestamped_filename(log_file_inc_training)

    @staticmethod
    def log_to_csv(file_path, number_of_classes, epoch, accuracy, mode='a'):
        #file_path = Logger.get_timestamped_filename(Logger.log_file_train)
        
        local_rank = 0
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if local_rank == 0:
            fieldnames = ['number_of_classes', 'epoch', 'accuracy']
            os.makedirs(os.path.dirname(Logger.file_name_normal), exist_ok=True)
            file_exists = os.path.isfile(Logger.file_name_normal)
            
            with open(Logger.file_name_normal, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w' or not file_exists:
                    writer.writeheader()
                writer.writerow({'number_of_classes': number_of_classes, 'epoch': epoch, 'accuracy': accuracy})
    
    @staticmethod
    def log_to_csv_incs_iterations(file_path, samples, iterations, mode='a'):
        #file_path = Logger.get_timestamped_filename(Logger.log_file_inc_training)
        
        local_rank = 0
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if local_rank == 0:
            fieldnames = ['Classes', 'iterations']
            Logger.print_with_timestamp(f"Logging: samples={samples}, iterations={iterations}")
            
            os.makedirs(os.path.dirname(Logger.file_name_inc), exist_ok=True)
            file_exists = os.path.isfile(Logger.file_name_inc)
            
            with open(Logger.file_name_inc, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w' or not file_exists:
                    writer.writeheader()
                writer.writerow({'Classes': samples, 'iterations': iterations})

builtins.print = Logger.print_with_timestamp

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} EXECUTED in {end_time - start_time:.5f} seconds")
        return result
    return wrapper