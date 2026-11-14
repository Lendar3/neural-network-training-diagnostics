from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import main
from src.neural_network import evaluate_neural_network
from src.logistic_regression import evaluate_logistic_regression
OUT=Path(__file__).resolve().parent
if __name__=='__main__':
    evidence=json.loads((OUT/'experiment_results.json').read_text())
    networks=[r for r in evidence['runs'] if r['model']=='neural_network']
    winner=min(networks,key=lambda r:r['validation_bce'])
    baseline=next(r for r in evidence['runs'] if r['model']=='logistic_regression')
    selection={'criterion':'minimum validation BCE','selected_weights':winner['weights'],
               'learning_rate':winner['learning_rate'],'best_epoch':winner['best_epoch'],
               'validation_bce':winner['validation_bce'],'baseline_weights':baseline['weights'],
               'note':'Selection saved before this final test call. Historical test exposure disclosed in report.'}
    (OUT/'selection_before_test.json').write_text(json.dumps(selection,indent=2))
    with np.load(OUT/'selected_data.npz') as data:
        X_test=data['X_test'];Y_test=data['Y_test']
    test=[]
    for row,evaluate in [(winner,evaluate_neural_network),(baseline,evaluate_logistic_regression)]:
        path=ROOT/row['weights']
        metrics=evaluate(path,X_test,Y_test)
        main.save_result(path,row['model'],'test',metrics)
        test.append({'model':row['model'],'weights':row['weights'],'rows':len(X_test),**metrics})
    evidence['selection']=selection
    evidence['test']=test
    (OUT/'experiment_results.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print('FINAL EVALUATION COMPLETE')
