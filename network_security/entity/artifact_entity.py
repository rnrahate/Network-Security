from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path: str
    train_path: str
    test_path: str