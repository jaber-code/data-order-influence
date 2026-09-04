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

        self.similarity_method = None


    def _load_config(self, file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def get_attribute(self, key):
        return self.config.get(key)

    def get_similarity_method(self):
        if self.similarity_method is None:
            self.similarity_method = self.config.get("similarity_method")
        return self.similarity_method