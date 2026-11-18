"""Fit existing GP once per task; CDF and derived EI evaluator are distinct."""
import numpy as np
from models.gp import GP
from .protocol import methods
from .probability import gaussian
from .metrics import sample,task_loss


def fit(task,track,c,checkpoint_path=None):
    X=np.array(task['raw']['X']);y=np.array(task['raw']['y']);tr=task['split']['train'];va=task['split']['validation']
    model=GP(c['dataset']['domain'],seed=task['streams']['gp'],**c['GP']).fit(X[tr],y[tr])
    mean,variance=model.posterior(X[va],include_observation_noise=False)
    lookup={rid:(float(mu),float(v)) for rid,mu,v in zip(va,mean,variance)};predictions=[]
    for row in task['tracks'][track]['validation']:
        mu,v=lookup[row['point_index']];s=gaussian(mu,v,task['constants']['tau'],task['constants']['U_train'],c)
        prefix='event' if track=='A' else 'augmented';lp,lm=s[prefix+'_log_p'],s[prefix+'_log_not_p']
        rid=row['point_index'];predictions.append(dict(**row,x=float(X[rid,0]),y=float(y[rid]),tau=task['constants']['tau'],U_train=task['constants']['U_train'],posterior_mean=mu,latent_variance=v,**s,
            probability=s[prefix+'_probability'],log_p=lp,log_not_p=lm,**sample(lp,lm,row['label'],row['weight'])))
    result=dict(method=methods(c,track)[1],model_fits=1,model_seed=task['streams']['gp'],
        training_X=X[tr].tolist(),training_y=y[tr].tolist(),GP_options=model.options,diagnostics=model.diagnostics,
        include_observation_noise=False,class_support=dict(training_raw_event_positive=int(np.sum(y[tr]<task['constants']['tau'])),training_raw_event_negative=int(np.sum(y[tr]>=task['constants']['tau'])),validation_positive=sum(r['label'] for r in predictions),validation_negative=sum(r['label']==0 for r in predictions)),predictions=predictions,task_loss=task_loss(predictions))
    if checkpoint_path is not None:
        from .checkpoints import save_estimator
        result['model_checkpoint']=save_estimator(checkpoint_path,model.model,'gaussian_process')
    return result
