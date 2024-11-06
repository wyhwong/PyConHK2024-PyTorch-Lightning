import enum


class Phase(enum.StrEnum):
    """Enum for the different phases of the demo"""

    TRAINING = "train"
    VALIDATION = "valid"
    TESTING = "test"
