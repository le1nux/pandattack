class DatasetNotFoundError(Exception):
    """Exception raised when a dataset could not be found."""
    pass

class DatasetOutOfBoundsError(Exception):
    """Exception raised when an index >= len(Dataset) is used."""
    pass