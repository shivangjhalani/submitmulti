from typing import Optional

class MultiCoCoError(Exception):
    pass

class ConfigurationError(MultiCoCoError):
    pass

class ModelInitializationError(MultiCoCoError):
    pass

class DatasetError(MultiCoCoError):
    pass

class DataLoadingError(DatasetError):
    pass

class ImageProcessingError(DatasetError):
    pass

class EvaluationError(MultiCoCoError):
    pass

class AnswerExtractionError(EvaluationError):
    pass

class DtypeMismatchError(MultiCoCoError):

    def __init__(self, expected_dtype: str, actual_dtype: str, message: Optional[str]=None):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        if message is None:
            message = f'Dtype mismatch: expected {expected_dtype}, got {actual_dtype}'
        super().__init__(message)