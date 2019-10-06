class DatasetNotFoundError(Exception):
    """Exception raised when a dataset could not be found."""
    pass

class DatasetOutOfBoundsError(Exception):
    """Exception raised when an index >= len(Dataset) is used."""
    pass

class AttackError(Exception):
    """Exception raised when there was a general issue with generation an adversarial example."""
    pass

class AdversarialNotFoundError(Exception):
    """Exception raised when an attack was not able to provide an adversarial example."""
    pass

class ModelNotTrainedError(Exception):
    """Exception raised when model state is requested that is training dependent"""
    pass


class DatasetFileCorruptError(Exception):
    """Thrown when integrity checks indicate that a given file is corrupt."""
    pass