"""Native RF1000 and existing LFBO EI recipe on explicitly prepared rows."""
import copy
import numpy as np
from models.lfbo import LFBO,RandomForestClassifier
from .protocol import require,methods
from .probability import rf_scores
from .metrics import sample,task_loss


def fit(task,track,c,checkpoint_path=None):
    rows=task['tracks'][track];tr=rows['training'];va=rows['validation'];X=np.array(task['raw']['X'])
    stream='rf-event' if track=='A' else 'lfbo-ei'
    owner=LFBO(c['dataset']['domain'],seed=task['streams'][stream],**c['RF']['owner_options'])
    model=RandomForestClassifier(random_state=owner.classifier_seed,**c['RF']['parameters'])
    z=[r['label'] for r in tr];require(set(z)=={0,1},'RF training class support')
    model.fit(X[[r['point_index'] for r in tr]],z,sample_weight=[r['weight'] for r in tr])
    columns=np.flatnonzero(model.classes_==1);require(len(columns)==1,'Missing RF positive class')
    probabilities=model.predict_proba(X[[r['point_index'] for r in va]])[:,columns[0]]
    predictions=[]
    for row,p in zip(va,probabilities):
        scores=rf_scores(float(p),c['classification']['RF_epsilon'])
        rid=row['point_index'];predictions.append(dict(**row,x=task['raw']['X'][rid][0],y=task['raw']['y'][rid],tau=task['constants']['tau'],U_train=task['constants']['U_train'],**scores,**sample(scores['log_p'],scores['log_not_p'],row['label'],row['weight'])))
    result=dict(method=methods(c,track)[0],model_fits=1,model_seed=task['streams'][stream],classifier_seed=owner.classifier_seed,
        RF_parameters=model.get_params(),classes=model.classes_.tolist(),positive_column=int(columns[0]),
        estimator_random_states=[v.random_state for v in model.estimators_],owner_RNG_after=copy.deepcopy(owner.rng.bit_generator.state),
        training_X=X[[r['point_index'] for r in tr]].tolist(),training_labels=z,training_weights=[r['weight'] for r in tr],
        class_support=dict(training_positive=sum(z),training_negative=len(z)-sum(z),validation_positive=sum(r['label'] for r in va),validation_negative=sum(r['label']==0 for r in va)),
        predictions=predictions,task_loss=task_loss(predictions))
    if checkpoint_path is not None:
        from .checkpoints import save_estimator
        result['model_checkpoint']=save_estimator(checkpoint_path,model,'random_forest')
    return result
