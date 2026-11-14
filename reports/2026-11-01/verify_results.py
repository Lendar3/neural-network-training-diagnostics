from pathlib import Path
import json, hashlib, ast
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
e=json.loads((OUT/'experiment_results.json').read_text())
checks=[]
with np.load(OUT/'selected_data.npz') as data:
    X_train=data['X_train']; y_train=data['Y_train'].ravel()
    X_val=data['X_validation']; y_val=data['Y_validation'].ravel()
    X_test=data['X_test']; y_test=data['Y_test'].ravel()
for path,expected in e['code_sha256'].items():
    snapshot = OUT/'source_snapshot'/Path(path).name
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==expected, path
    assert ast.dump(ast.parse(snapshot.read_text())) == ast.dump(ast.parse((ROOT/path).read_text())), path
checks.append('Source snapshots match run hashes; current executable AST identical (comments may differ).')
scalers=[]
for r in e['runs']:
    with np.load(ROOT/r['weights']) as d:
        scalers.append((d['scaler_mean'],d['scaler_scale']))
        np.testing.assert_allclose(d['scaler_mean'],X_train.mean(axis=0,dtype=np.float64),atol=1e-10)
        X=((X_val-d['scaler_mean'])/d['scaler_scale'])
        if r['model']=='logistic_regression':
            z=(X@d['coef'].T+d['intercept']).ravel()
        else:
            # Independent NumPy forward pass through saved Linear/ReLU layers.
            X=X.astype(np.float32)
            for layer in ['0','2']:
                X=np.maximum(X@d[layer+'.weight'].T+d[layer+'.bias'],0)
            z=(X@d['4.weight'].T+d['4.bias']).ravel()
            assert int(d['best_epoch'])==np.argmin(d['validation_losses'])+1
            assert len(d['train_losses'])==300
        p=expit(z);pred=p>=.5
        precision,recall,f1,_=precision_recall_fscore_support(y_val,pred,average='binary',zero_division=0)
        values={'bce':np.mean(np.logaddexp(0,z)-y_val*z),'accuracy':accuracy_score(y_val,pred),
                'precision':precision,'recall':recall,'f1':f1,'roc_auc':roc_auc_score(y_val,p)}
        for k,v in values.items():np.testing.assert_allclose(v,r['validation_'+k],atol=2e-6,rtol=1e-6)
for mean,scale in scalers[1:]:
    np.testing.assert_array_equal(mean,scalers[0][0]);np.testing.assert_array_equal(scale,scalers[0][1])
checks.append('All four validation metric sets reproduced from saved weights using NumPy; scalers identical and fit only on train.')
confusions={}
for r in e['test']:
    with np.load(ROOT/r['weights']) as d:
        X=(X_test-d['scaler_mean'])/d['scaler_scale']
        if r['model']=='logistic_regression':z=(X@d['coef'].T+d['intercept']).ravel()
        else:
            X=X.astype(np.float32)
            for layer in ['0','2']:X=np.maximum(X@d[layer+'.weight'].T+d[layer+'.bias'],0)
            z=(X@d['4.weight'].T+d['4.bias']).ravel()
        p=expit(z);pred=p>=.5
        precision,recall,f1,_=precision_recall_fscore_support(y_test,pred,average='binary',zero_division=0)
        values={'bce':np.mean(np.logaddexp(0,z)-y_test*z),'accuracy':accuracy_score(y_test,pred),
                'precision':precision,'recall':recall,'f1':f1,'roc_auc':roc_auc_score(y_test,p)}
        for k,v in values.items():np.testing.assert_allclose(v,r[k],atol=2e-6,rtol=1e-6)
        confusions[r['model']]=confusion_matrix(y_test,pred,labels=[1,0]).tolist()
checks.append('Test metrics independently recomputed for verification only; no additional selection or training.')
html=(OUT/'report_ru.html').read_text();md=(OUT/'report_ru.md').read_text()
assert html.count('data:image/png;base64,')==7
for name in ['overfitting','learning_rates','validation_comparison','network_test_confusion_matrix','baseline_test_confusion_matrix','network_test_roc_curve','baseline_test_roc_curve']:
    assert (OUT/'assets'/f'{name}.png').is_file()
checks.append('Seven local figures exist and are embedded in self-contained HTML.')
(OUT/'verification.json').write_text(json.dumps({'checks':checks,'test_confusion_matrices_order_1_0':confusions},indent=2))
print('\n'.join(checks))
