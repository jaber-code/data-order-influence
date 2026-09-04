from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class SimilarityBase(ABC):

    @abstractmethod
    def get_similarity(self, word1, word2):
        pass


class SimilarityBERT(SimilarityBase):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.pipeline = pipeline('feature-extraction', model='bert-base-uncased')

    def get_similarity(self, word1, word2):
        encoded_input1 = self.tokenizer(word1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_input2 = self.tokenizer(word2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            output1 = self.model(**encoded_input1)
            output2 = self.model(**encoded_input2)

        cls_embedding1 = output1.last_hidden_state[:, 0, :]
        cls_embedding2 = output2.last_hidden_state[:, 0, :]

        similarity = F.cosine_similarity(cls_embedding1, cls_embedding2).item()
        #print("Compare ", word1, " with ", word2, "  sim= ", similarity)
        return similarity


class SimilarityNLTK(SimilarityBase):
    def __init__(self):
        print("Downloading WordNet data...")
        nltk.download('wordnet')

    def _get_lch_similarity(self, word1, word2):
        synset1 = wn.synsets(word1)
        synset2 = wn.synsets(word2)

        if synset1 and synset2:
            similarity = synset1[0].lch_similarity(synset2[0])
            if similarity is not None:
                return similarity
        return 0

    def _get_path_similarity(self, word1, word2):
        synset1 = wn.synsets(word1)
        synset2 = wn.synsets(word2)

        if synset1 and synset2:
            similarity = synset1[0].path_similarity(synset2[0])
            #print("Compare ", word1, " with ", word2, "  sim= ", similarity)
            if similarity is not None:
                return similarity
        return 0

    def _compare_sentences(self, sentence1, sentence2):
        words1 = word_tokenize(sentence1)
        words2 = word_tokenize(sentence2)

        words1 = [word.lower() for word in words1 if word.isalnum()]
        words2 = [word.lower() for word in words2 if word.isalnum()]

        total_similarity = 0
        num_pairs = 0

        for word1 in words1:
            for word2 in words2:
                similarity = self._get_path_similarity(word1, word2)
                total_similarity += similarity
                num_pairs += 1

        if num_pairs == 0:
            return 0

        return total_similarity / num_pairs

    def get_similarity(self, word1, word2):
        return self._compare_sentences(word1, word2)
    
class SimilaritySBERT(SimilarityBase):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def _compare_sentences(self, sentence1, sentence2):
        embedding1 = self.model.encode(sentence1, convert_to_tensor=True)
        embedding2 = self.model.encode(sentence2, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    
    def get_similarity(self, sentence1, sentence2):
        return self._compare_sentences(sentence1, sentence2)
    
"""
if __name__ == "__main__":
    print("12")
    sbert = SimilaritySBERT()

    sim1 = sbert.get_similarity('hamster', 'tabby')
    print('sim1 ', sim1)

    sim2 = sbert.get_similarity('hamster', 'tabby, tabby cat')
    print('sim2 ', sim2)

    sim3 = sbert.get_similarity('A hamster is a small, typically nocturnal rodent, often kept as a pet.'
                                ,'A tabby cat is a domestic cat with a distinctive M-shaped marking on its forehead, typically characterized by a mix of brown and grey stripes or dots.')
    print('sim3 ', sim3)
"""

"""
Expand each of the following terms with 1-2 sentences of strictly relevant contextual information. Follow these rules:
1. Focus only on factual, domain-specific context
2. Omit all introductory phrases like "This is..." or "The term refers to..."
3. Preserve the original term's case and formatting
4. Prioritize scientific/technical context for technical terms

"""

