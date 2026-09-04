import random
import os
import heapq
from similarity_factory import *
from data_order_core.config.config_manager import ConfigManager
from enums import *
import requests

class ImageNetClassProcessor:
    _disable_alph_generator = False
    _disable_rand_generator = False
    _disable_mostdiss_generator = False
    _disable_extend_words_generator = True

    _PATH_IMAGENET_CLASSES_NAMES = 'classes_order_generator/imagenet_metadata.txt'
    _PATH_IMAGENET_CLASSES_IDS = '/fscratch/sjaber/imagenet/train/'
    _PATH_IMAGENET_CLASSES_IDS_LOCAL = 'classes_order_generator/imagenet_class_ids.txt'

    _LLM_LINK_LOCAL = 'meta-llama/Llama-2-7b-chat-hf'
    _LLM_LINK = 'http://serv-3306.kl.dfki.de:8000/v1'


    def __init__(self, config_path="data_order_core/config/config.yaml"):  
        self._config = ConfigManager(config_path)

        sim_method_str = self._config.get_similarity_method()
        sim_method_type = SimilarityMethod(sim_method_str)
        self._similarity_module = SimilarityCalculatorSingleton(method=sim_method_type)

    def generate_extended_definitions(self, classes):
        print("execute: generate_extended_definitions")

        print("classes, ", classes[:10])
        API_URL = "http://serv-3306.kl.dfki.de:8000/v1/chat/completions"
        headers = {"Authorization": "Bearer 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"}

        base_prompt = "Write a definition combining these words in the right context, don't add any unneccessary or irrelevant words nor introductions. Words: "

        output_file = "extended_names.txt"

        with open(output_file, 'w') as f:
            for words in classes:
                print("words: ", words)
                prompt = base_prompt + words

                payload = {"model": "meta-llama-3.1-70b-instruct-fp8", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}

                response = requests.post(API_URL, headers=headers, json=payload)
                generated_text = response.json()['choices'][0]['message']['content']
                f.write(words.split(",")[0] + ":" + generated_text+'\n')
                print("generated_text: ", generated_text)
                print("words.split(",")[0]: ", words.split(",")[0])


    def _sim(self, words1, words2, data_dict):
        print("words1 ", words1)
        print("words2 ", words2)

        def1 = data_dict[words1.split(",")[0]]
        def2 = data_dict[words2.split(",")[0]]

        print("def1: ", def1)
        print("def2: ", def2)

        s = self._similarity_module.get_similarity(def1, def2)
        print("sim= ", s)
        return s

    def _write_alphabetical_order(self, classes, output_file):
        print("execute: _write_alphabetical_order")

        with open(output_file, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")

    def _write_random_order(self, classes, output_file):
        print("execute: _write_random_order")

        shuffled_classes = classes.copy()
        random.shuffle(shuffled_classes)
        with open(output_file, 'w') as f:
            for class_name in shuffled_classes:
                f.write(f"{class_name}\n")

    def _write_most_dissimilar_first(self, classes, output_file):
        print("execute: _write_most_dissimilar_first")

        chosen_classes = [classes[0]]
        remaining_classes = set(classes[1:])

        file_path = "extended_names.txt"
        data_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                name, definition = line.strip().split(":", 1)
                data_dict[name] = definition

        print(data_dict)

        min_similarity = {cls: self._sim(cls, chosen_classes[0], data_dict) for cls in remaining_classes}
        heap = [(min_similarity[cls], cls) for cls in remaining_classes]
        heapq.heapify(heap)

        while remaining_classes:
            while heap:
                _, candidate = heapq.heappop(heap)
                if candidate in remaining_classes:
                    break

            chosen_classes.append(candidate)
            remaining_classes.remove(candidate)

            for cls in remaining_classes:
                min_similarity[cls] = min(min_similarity[cls], self._sim(cls, candidate, data_dict))
                heapq.heappush(heap, (min_similarity[cls], cls))

        with open(output_file, 'w') as f:
            for class_name in chosen_classes:
                f.write(f"{class_name}\n")

    def generate(self, classes_names):

        if not ImageNetClassProcessor._disable_extend_words_generator:
            self.generate_extended_definitions(classes_names)

        if not ImageNetClassProcessor._disable_alph_generator:
            self._write_alphabetical_order(classes_names, "generated_files/alphabetical_order.txt")

        if not ImageNetClassProcessor._disable_rand_generator:
            self._write_random_order(classes_names, "generated_files/random_order.txt")

        if not ImageNetClassProcessor._disable_mostdiss_generator:
            self._write_most_dissimilar_first(classes_names, "generated_files/most_dissimilar_first.txt")




    def _get_classes_ids(self, directory):
        return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    def read_classes_from_file(self):
        classes = []
        try:
            class_ids = set(self._get_classes_ids(self._PATH_IMAGENET_CLASSES_IDS))
            self._PATH_IMAGENET_CLASSES_NAMES = '/ds/images/imagenet/imagenet_metadata.txt'
        except Exception as e:
            print("Local mode:", e)
            class_ids = set()
            with open(self._PATH_IMAGENET_CLASSES_IDS_LOCAL, 'r') as file:
                for line in file:
                    class_id = line.strip()
                    if class_id:
                        class_ids.add(class_id)

        with open(self._PATH_IMAGENET_CLASSES_NAMES, 'r') as file:
            for i, line in enumerate(file):
                if i % 10000 == 0:
                    print(f"Processing line {i}...")
                parts = line.strip().split('\t')
                if len(parts) > 1 and parts[0] in class_ids:
                    classes.append(parts[1])

        return classes

    def main(self):
        try:
            classes = self.read_classes_from_file()
            self.generate(classes)
        except Exception as e:
            print(f"Error reading classes: {e}")
        

if __name__ == '__main__':
    processor = ImageNetClassProcessor()
    try:
        processor.main()
    except Exception as e:
        print(f"Error: {e}")
