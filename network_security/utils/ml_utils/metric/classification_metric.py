import sys
from network_security.entity.artifact_entity import ClassificationMetricArtifact
from network_security.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        f1 = float(f1_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred))
        accuracy = float(accuracy_score(y_true, y_pred))

        classification_metric = ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall,
            accuracy_score=accuracy
        )
        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e, sys.exc_info()) from e