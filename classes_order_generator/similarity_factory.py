from enums import *
from similarity_models import *

class SimilarityCalculatorSingleton:
    _instance = None

    def __new__(cls, method=SimilarityMethod.BERT) -> SimilarityBase:
        if cls._instance is None:
            cls._instance = SimilarityFactory.create_similarity(method)
        return cls._instance
    

class SimilarityFactory:

    @staticmethod
    def create_similarity(method: SimilarityMethod) -> SimilarityBase:

        if method == SimilarityMethod.BERT:
            return SimilarityBERT()
        elif method == SimilarityMethod.NLTK:
            return SimilarityNLTK()
        elif method == SimilarityMethod.S_BERT:
            return SimilaritySBERT()
        else:
            raise ValueError(f"Unsupported similarity method: {method}")