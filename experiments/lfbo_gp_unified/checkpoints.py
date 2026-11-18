"""Save native fitted estimators; ordinary verification only checks bytes."""
from pathlib import Path
import os
import uuid

from .common import require, sha


def selected(policy, family_index, task_index):
    require(policy in ('none', 'representative', 'all'), 'Unknown checkpoint policy')
    return policy == 'all' or (policy == 'representative' and family_index == task_index == 0)


def model_path(directory, task_id, method):
    return Path(directory) / 'models' / task_id / (method + '.joblib')


def save_estimator(path, estimator, kind):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        joblib.dump(estimator, temp, compress=3)
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
        # Called only before the unit receipt exists. A previous interrupted
        # attempt may leave an uncommitted checkpoint; completed units are reused.
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return dict(format='joblib-native-estimator-v1', kind=kind, sha256=sha(path))


def check(path, receipt):
    require(receipt['format'] == 'joblib-native-estimator-v1', 'Unknown checkpoint format')
    require(Path(path).is_file() and sha(path) == receipt['sha256'],
            'Model checkpoint missing or changed: ' + str(path))
