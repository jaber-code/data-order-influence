import yaml

# How to define Signleton: https://medium.com/analytics-vidhya/how-to-create-a-thread-safe-singleton-class-in-python-822e1170a7f6
class ConfigManager:
    _instance = None

    def __new__(cls, file_path=None):
        if cls._instance is None: 
                if not cls._instance:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialize(file_path)
        return cls._instance

    def _initialize(self, file_path):
        if file_path:
            self.config = self._load_config(file_path)
        else:
            self.config = {}
        self.selected_model = None
        self.select_dataset_type = None
        self.epochs = None
        self.is_local = None
        self.num_classes = None
        self.use_optimizations = None
        self.similarity_method = None
        self.imagenet_path = None
        self.initial_num_classes = None
        self.step_num_classes = None
        self.use_multi_processing = None
        self.batch_size = None
        self.perform_inc_training = None
        self.perform_normal_training = None
        self.inc_training_type = None
        self.learning_rate = None
        self.inc_decay_type = None

    def _load_config(self, file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def get_attribute(self, key):
        return self.config.get(key)

    def get_selected_model(self):
        if self.selected_model is None:
            self.selected_model = self.config.get("selected_model")
        return self.selected_model

    def get_select_dataset_type(self):
        if self.select_dataset_type is None:
            self.select_dataset_type = self.config.get("select_dataset_type")
        return self.select_dataset_type

    def get_epochs(self):
        if self.epochs is None:
            self.epochs = int(self.config.get("epochs"))
        return self.epochs

    def get_is_local(self):
        if self.is_local is None:
            self.is_local = self.config.get("is_local")
        return self.is_local

    def get_num_classes(self):
        if self.num_classes is None:
            self.num_classes = self.config.get("num_classes")
        return self.num_classes

    def get_use_optimizations(self):
        if self.use_optimizations is None:
            self.use_optimizations = self.config.get("use_optimizations")
        return self.use_optimizations

    def get_similarity_method(self):
        if self.similarity_method is None:
            self.similarity_method = self.config.get("similarity_method")
        return self.similarity_method

    def get_imagenet_path(self):
        if self.imagenet_path is None:
            self.imagenet_path = self.config.get("imagenet_path")
        return self.imagenet_path

    def get_initial_num_classes(self):
        if self.initial_num_classes is None:
            self.initial_num_classes = self.config.get("initial_num_classes")
        return self.initial_num_classes

    def get_step_num_classes(self):
        if self.step_num_classes is None:
            self.step_num_classes = self.config.get("step_num_classes")
        return self.step_num_classes

    def get_use_multi_processing(self):
        if self.use_multi_processing is None:
            self.use_multi_processing = self.config.get("use_multi_processing")
        return self.use_multi_processing
    
    def get_batch_size(self):
        if self.batch_size is None:
            self.batch_size = self.config.get("batch_size")
        return self.batch_size

    def get_perform_inc_training(self):
        if self.perform_inc_training is None:
            self.perform_inc_training = self.config.get("perform_inc_training")
        return self.perform_inc_training
    
    def get_perform_normal_training(self):
        if self.perform_normal_training is None:
            self.perform_normal_training = self.config.get("perform_normal_training")
        return self.perform_normal_training
    
    def get_inc_training_type(self):
        if self.inc_training_type is None:
            self.inc_training_type = self.config.get("inc_training_type")
        return self.inc_training_type
    
    def get_learning_rate(self):
        if self.learning_rate is None:
            self.learning_rate = self.config.get("learning_rate")
        return self.learning_rate
    
    def get_inc_decay_type(self):
        if self.inc_decay_type is None:
            self.inc_decay_type = self.config.get("inc_decay_type")
        return self.inc_decay_type
