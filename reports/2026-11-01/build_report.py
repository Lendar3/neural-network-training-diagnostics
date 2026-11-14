"""Собрать отчёт и картинки из сохранённых результатов, без нового обучения."""
from pathlib import Path
import json,csv,html,base64,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ASSETS=OUT/'assets';ASSETS.mkdir(exist_ok=True)
e=json.loads((OUT/'experiment_results.json').read_text())
runs=e['runs'];nets=[r for r in runs if r['model']=='neural_network']
base=next(r for r in runs if r['model']=='logistic_regression')
small_base,small=e['small_runs']
winner=min(nets,key=lambda r:r['validation_bce'])
test_net,test_base=e['test']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,
                     'grid.alpha':.2,'savefig.dpi':170})
blue='#2166ac';orange='#b35806'
with np.load(ROOT/small['weights']) as d:
    train=d['train_losses'];val=d['validation_losses'];best=int(d['best_epoch'])
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for ax,limit,title in zip(axes,[1000,150],['Весь запуск: 1000 эпох','Начало обучения: первые 150 эпох']):
    epochs=np.arange(1,limit+1)
    ax.plot(epochs,train[:limit],color=blue,label='Train BCE')
    ax.plot(epochs,val[:limit],color=orange,label='Validation BCE')
    ax.axvline(best,color='#555555',ls=':',label=f'Checkpoint: эпоха {best}')
    ax.scatter([best],[val[best-1]],color=orange,zorder=4)
    ax.set(xlabel='Эпоха',ylabel='Средняя BCE',title=title);ax.legend(fontsize=9)
fig.savefig(ASSETS/'overfitting.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True,layout='constrained')
for ax,r in zip(axes,nets):
    with np.load(ROOT/r['weights']) as d: a=d['train_losses'];b=d['validation_losses']
    ax.plot(np.arange(1,301),a,color=blue,label='Train BCE')
    ax.plot(np.arange(1,301),b,color=orange,label='Validation BCE')
    ax.axhline(base['validation_bce'],color='#666666',ls='--',label='Baseline: validation BCE')
    ax.set(xlabel='Эпоха',title=f"LR = {r['learning_rate']:g}\nЛучшая validation BCE = {r['validation_bce']:.4f}",ylim=(.49,.71))
    ax.legend(fontsize=8,loc='lower left')
axes[0].set_ylabel('Средняя BCE')
fig.savefig(ASSETS/'learning_rates.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
labels=['Baseline','LR=0.0001','LR=0.001','LR=0.01']
values=[base['validation_bce']]+[r['validation_bce'] for r in nets]
bars=ax.barh(labels,values,color=['#777777',blue,blue,orange])
ax.bar_label(bars,fmt='%.4f',padding=5)
ax.set(xlim=(0,.76),xlabel='Минимальная validation BCE — меньше лучше',title='Одинаковые данные; сети обучались по 300 эпох')
ax.invert_yaxis();fig.savefig(ASSETS/'validation_comparison.png');plt.close(fig)
for r,folder,prefix in [(test_net,'bce','network'),(test_base,'logistic_regression','baseline')]:
    stem=Path(r['weights']).stem
    for suffix in ['test_confusion_matrix','test_roc_curve']:
        shutil.copy2(ROOT/'img'/folder/f'{stem}_{suffix}.png',ASSETS/f'{prefix}_{suffix}.png')

md=[];parts=[]
def heading(text,level=2):
    md.append('#'*level+' '+text+'\n');parts.append(f'<h{level}>{html.escape(text)}</h{level}>')
def para(text):
    md.append(text+'\n');parts.append('<p>'+html.escape(text)+'</p>')
def table(headers,rows):
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
    parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def figure(name,caption):
    md.append(f'![{caption}](assets/{name})\n\n{caption}\n')
    encoded=base64.b64encode((ASSETS/name).read_bytes()).decode()
    parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
def f(x):return f'{x:.4f}'
def pct(x):return f'{100*x:.2f}%'

heading('Project 1. Diagnosing neural network training',1)
para('HIGGS · отчёт по экспериментам · 1 ноября 2026 года')
heading('Что получилось')
para(f"Показаны две проблемы: переобучение на 300 train-примерах и недостаточное обучение при маленьком learning rate за фиксированный бюджет. В малом опыте возврат к эпохе {small['best_epoch']} уменьшил validation BCE с {f(small['validation_bce_last'])} до {f(small['validation_bce'])}. Это улучшение относительно последних весов, но не победа над baseline: у логистической регрессии validation BCE {f(small_base['validation_bce'])}.")
para(f"В контролируемом сравнении на 100 000 train-примерах лучший из трёх learning rate — 0.01: validation BCE {f(winner['validation_bce'])}, F1 {f(winner['validation_f1'])}. После выбора по validation сеть получила test accuracy {pct(test_net['accuracy'])}, baseline — {pct(test_base['accuracy'])}; разница {(test_net['accuracy']-test_base['accuracy'])*100:.2f} процентного пункта. Вывод ограничен этим разбиением, seed=12 и бюджетом 300 эпох.")
heading('1. Задача, данные и правила сравнения')
para('Задача — бинарная классификация событий HIGGS по 28 признакам: класс 1 — signal, класс 0 — background. Baseline — логистическая регрессия. Основная модель — многослойная сеть 28 → 32 → ReLU → 16 → ReLU → 1. В сети 1473 обучаемых параметра. Последний выход — логит; для метрик он преобразуется sigmoid в вероятность класса 1.')
para('Весь исходный набор содержит 11 000 000 событий. Первое разбиение train/validation/test — 70/15/15 со стратификацией и seed=12. Затем текущая программа берёт первые rows train-строк и пропорциональные срезы validation/test. Эти уменьшенные срезы не стратифицируются повторно. Все выбранные значения проверены на конечность; в каждом разделе присутствуют оба класса.')
table(['Опыт','Train','Validation','Test','Класс 1: train / val / test'],[
['Малый набор','300','64','64','156 / 33 / 30'],
['Сравнение LR','100 000','21 429','21 429','53 011 / 11 281 / 11 307']])
para('StandardScaler обучается только на выбранном train. К validation/test применяются его сохранённые средние и масштабы. Baseline и сети одного размера получают одинаковые строки. Для сравнения LR совпадение scaler между запусками дополнительно проверяется. Малый и большой опыты используют разные validation-наборы: их числа нельзя считать чистым эффектом увеличения train.')
para('Сеть обучается full-batch Adam: одна эпоха — одно обновление на всём train. Seed модели — 12; dropout и weight decay отсутствуют. Baseline: LogisticRegression(max_iter=1000, random_state=12), стандартный L-BFGS и регуляризация. Его max_iter — не число эпох нейросети.')
para('Главный критерий выбора — минимальная validation BCE. BCE измеряет ошибку вероятностей: меньше лучше. Accuracy показывает долю верных классов; precision, recall и F1 относятся к положительному классу 1 при пороге 0.5; ROC-AUC оценивает ранжирование вероятностей. Все показатели считаются для одного выбранного checkpoint, а не для разных лучших эпох по каждой метрике.')
heading('2. Переобучение на маленьком наборе')
para('Источник этого опыта — запуск пользователя и его сохранённые NPZ, обучение повторно не запускалось. Настройки: rows=300, LR=0.001, 1000 эпох, seed=12. Объяснение ниже ретроспективное: результат уже был известен из присланного лога. Нельзя представлять его как гипотезу, впервые записанную до этого запуска.')
para('Гипотеза: при длительном обучении на маленьком наборе сеть подстраивается под особенности 300 примеров, которые не помогают новым событиям. Ожидаемый признак — train loss продолжает снижаться, а validation loss после минимума растёт. Отсутствие регуляризации и малый train делают такое объяснение правдоподобным, но эксперимент не выделяет вклад каждой причины отдельно.')
table(['Состояние сети','Эпоха','Train BCE','Validation BCE'],[
['Лучший checkpoint',small['best_epoch'],f(small['train_bce_at_best']),f(small['validation_bce'])],
['Конец обучения',small['epochs'],f(small['train_bce_last']),f(small['validation_bce_last'])]])
figure('overfitting.png','Рисунок 1. Train BCE продолжает снижаться, а validation BCE после эпохи 86 растёт. Справа — увеличение начала того же запуска, не второй эксперимент.')
para(f"Измеренное исправление: после всех 1000 эпох код загрузил веса эпохи 86, выбранные по минимальной validation BCE. Ошибка validation уменьшилась на {small['validation_bce_last']-small['validation_bce']:.4f}, с {f(small['validation_bce_last'])} до {f(small['validation_bce'])}. Механизм исправления — отказ от последних, переобученных весов в пользу более раннего checkpoint.")
para('Это не early stopping: обучение не прекращалось на эпохе 86. Восстановление уменьшает validation loss относительно последней эпохи, но само по себе не доказывает улучшение на независимых данных: checkpoint выбран именно по validation. Последние веса не сохранены, поэтому финальные F1/accuracy для эпохи 1000 здесь не выдумываются.')
table(['Модель, 300 train','Validation BCE','Accuracy','F1','ROC-AUC'],[
['Логистическая регрессия',f(small_base['validation_bce']),pct(small_base['validation_accuracy']),f(small_base['validation_f1']),f(small_base['validation_roc_auc'])],
['Сеть, checkpoint 86',f(small['validation_bce']),pct(small['validation_accuracy']),f(small['validation_f1']),f(small['validation_roc_auc'])]])
para('Вывод: характерное переобучение наблюдается, возврат checkpoint помогает по BCE, но на этом validation baseline немного лучше восстановленной сети. Всего 64 validation-примера: один верный ответ меняет accuracy на 1.5625 процентного пункта, поэтому небольшие различия нельзя объявлять устойчивым преимуществом.')
heading('3. Три learning rate при одинаковом бюджете')
para('Эта гипотеза записана до новых запусков в hypotheses_before_runs.md: LR=0.0001 может быть слишком мал для 300 full-batch обновлений; увеличение до 0.001 или 0.01 может уменьшить train и validation BCE. Менялся только LR. Фиксированы rows=100000, 300 эпох, архитектура, Adam, seed=12, preprocessing и разбиение. Scheduler, dropout и clipping не добавлялись.')
rows=[]
for r in nets:rows.append([f"{r['learning_rate']:g}",r['best_epoch'],f(r['train_bce_at_best']),f(r['validation_bce']),pct(r['validation_accuracy']),f(r['validation_f1']),f(r['validation_roc_auc'])])
table(['LR','Лучшая эпоха','Train BCE¹','Validation BCE','Accuracy','F1','ROC-AUC'],rows)
para('¹ Train BCE взята на той же лучшей эпохе, что и validation-метрики. Для LR=0.01 последняя эпоха 300 имеет validation BCE 0.5385; сохранена эпоха 299 с BCE 0.5384.')
figure('learning_rates.png','Рисунок 2. Одинаковая шкала BCE во всех панелях. Пунктир — validation BCE логистической регрессии на тех же данных.')
para(f"При LR=0.0001 train/validation BCE равны {f(nets[0]['train_bce_at_best'])}/{f(nets[0]['validation_bce'])}: обе ошибки выше, чем у более быстро обучившихся вариантов. При LR=0.01 они равны {f(winner['train_bce_at_best'])}/{f(winner['validation_bce'])}. Validation BCE уменьшилась на {nets[0]['validation_bce']-winner['validation_bce']:.4f}, accuracy выросла на {(winner['validation_accuracy']-nets[0]['validation_accuracy'])*100:.2f} процентного пункта.")
para('Вывод: гипотеза недостаточного обучения при маленьком LR поддерживается в пределах 300 обновлений. Архитектура не менялась, поэтому данные не доказывают, что она слишком маленькая. Более долгий запуск с LR=0.0001 мог бы улучшиться; этот опыт не определяет его предельное качество. Это underfitting из-за недостаточного продвижения оптимизации при заданном бюджете.')
para(f"У LR=0.01 разрыв validation−train BCE составляет {winner['validation_bce']-winner['train_bce_at_best']:.4f}. Это не повод заявлять полное отсутствие переобучения. В рамках 300 эпох нет роста validation, сопоставимого с малым опытом; дальнейшее увеличение эпох здесь не исследовано.")
heading('4. Сравнение с baseline и выбор конфигурации')
figure('validation_comparison.png','Рисунок 3. Минимальная validation BCE; столбцы начинаются с нуля. Baseline обучен на том же train, что и три сети.')
table(['Модель','Validation BCE','Accuracy','Precision','Recall','F1','ROC-AUC'],[
['Baseline',f(base['validation_bce']),pct(base['validation_accuracy']),f(base['validation_precision']),f(base['validation_recall']),f(base['validation_f1']),f(base['validation_roc_auc'])],
['Сеть LR=0.01',f(winner['validation_bce']),pct(winner['validation_accuracy']),f(winner['validation_precision']),f(winner['validation_recall']),f(winner['validation_f1']),f(winner['validation_roc_auc'])]])
para('У LR=0.0001 F1 чуть выше baseline, хотя BCE и ROC-AUC хуже. Это показывает, почему нельзя делать вывод по одному F1: модель может чаще выдавать класс 1 и повышать recall ценой precision. Критерий выбора не менялся после просмотра результатов — использовалась validation BCE.')
para(f"Выбрана сеть LR=0.01, 100000 train, 300 эпох, checkpoint {winner['best_epoch']}, seed=12, порог 0.5. Решение записано в selection_before_test.json до новой test-оценки. Это лучший проверенный вариант, а не доказанный глобальный оптимум.")
heading('5. Финальное сравнение на test')
para('После выбора конфигурации оценены только выбранная сеть и соответствующий baseline на одинаковых 21 429 test-примерах. В рамках этой серии test не использовался для выбора LR. Однако test уже просматривался в истории проекта: в присланном логе есть оценка малых моделей на первых 64 test-примерах, а в предыдущем аудите были другие test-оценки. Поэтому этот результат не называется проверкой на никогда не использованном holdout. После просмотра test новые настройки не подбирались.')
table(['Модель','Test BCE','Accuracy','Precision','Recall','F1','ROC-AUC'],[
['Baseline',f(test_base['bce']),pct(test_base['accuracy']),f(test_base['precision']),f(test_base['recall']),f(test_base['f1']),f(test_base['roc_auc'])],
['Выбранная сеть',f(test_net['bce']),pct(test_net['accuracy']),f(test_net['precision']),f(test_net['recall']),f(test_net['f1']),f(test_net['roc_auc'])]])
para(f"Сеть лучше baseline на этом test: BCE ниже на {test_base['bce']-test_net['bce']:.4f}, accuracy выше на {(test_net['accuracy']-test_base['accuracy'])*100:.2f} п.п., F1 выше на {test_net['f1']-test_base['f1']:.4f}, ROC-AUC выше на {test_net['roc_auc']-test_base['roc_auc']:.4f}. Recall близок, а precision сети заметно выше: улучшение не объясняется только увеличением числа положительных предсказаний.")
figure('network_test_confusion_matrix.png','Рисунок 4. Выбранная сеть на test: строки — правильный класс, столбцы — предсказанный. Порядок классов [1, 0].')
figure('baseline_test_confusion_matrix.png','Рисунок 5. Baseline на том же test и с тем же порогом 0.5.')
figure('network_test_roc_curve.png','Рисунок 6. ROC выбранной сети на test; диагональ — ориентир случайного ранжирования.')
figure('baseline_test_roc_curve.png','Рисунок 7. ROC baseline на том же test.')
heading('6. Ограничения и честная интерпретация')
para('Все новые модели использовали один seed=12 по настройкам проекта. Лекция рекомендует несколько одинаковых seed для каждого варианта и среднее со стандартным отклонением. Здесь такая проверка не проводилась: статистическая устойчивость результата между инициализациями не установлена. Одинаковый seed также не гарантирует побитовое совпадение между устройствами и версиями библиотек.')
para('На 300 train-примерах validation очень мал, и выбор лучшей эпохи среди 1000 вариантов может подстроиться под его случайные особенности. Результаты малого и большого наборов нельзя напрямую сравнивать как эксперимент только с размером train, потому что validation/test тоже изменились.')
para('Длительности сохранены для воспроизводимости, но не используются для ранжирования моделей по скорости. Baseline и сеть замеряют немного разные этапы; первый запуск MPS может включать прогрев. 1000 итераций L-BFGS нельзя приравнивать к 1000 эпохам full-batch Adam.')
para('Не проверялись dropout, weight decay, другие оптимизаторы, batch size и третий тип проблемы — нестабильность. Для данного задания уже продемонстрированы две проблемы и сравнение трёх конфигураций одного фактора; эти дополнительные методы не выдаются за выполненные эксперименты.')
heading('7. Вывод для защиты проекта')
para('На маленьком наборе длительное обучение привело к падению train BCE почти до нуля и росту validation BCE. Это согласуется с запоминанием обучающих примеров. Возврат к checkpoint минимальной validation BCE уменьшил ошибку по сравнению с последними весами, но не сделал сеть лучше baseline на этом маленьком validation.')
para('Во втором опыте слишком маленький learning rate ограничивал обучение за 300 обновлений. Увеличение LR при неизменных остальных условиях уменьшило обе потери. Из трёх проверенных значений выбран LR=0.01 по validation BCE. Сеть с этим LR превзошла логистическую регрессию в финальном сравнении. Утверждение относится к одному разбиению и seed; история просмотра test и малый размер первого validation ограничивают силу выводов.')
heading('8. Источники, воспроизводимость и помощь AI')
para('Методика: локальная лекция 03 «Diagnosing the training process», разделы Controlled comparison, Analyzing results и Writing the experiment report; связанный lab_03_pytorch.ipynb. Правило аргументации: наблюдение → гипотеза → контроль условий → изменение одного фактора → повторное измерение. Условия Project 1 предоставлены пользователем в чате.')
para('Существующий малый опыт: user_run_log.txt и два NPZ с rows300. Новые измерения: new_runs.log, final_test.log, experiment_results.json и NPZ, перечисленные ниже. Числа отчёта извлечены из реальных файлов; ожидаемые результаты не подставлялись вместо измерений.')
para('Использование AI: Codex прочитал код и лог, записал гипотезу до новых LR-запусков, вызвал существующие функции обучения, собрал графики и этот текст. Codex не изменял основные main.py и src/*.py при выполнении данной серии. Сохранён снимок исходников на момент запусков; поздние изменения комментариев в рабочем файле сети не меняют вычисления. Выводы нужно уметь объяснить самостоятельно: особенно различие checkpoint selection/early stopping и validation/test. Первоначальная гипотеза малого опыта сформулирована ретроспективно.')
para('Окружение: Python '+e['python']+', '+e['platform']+', устройство '+e['device']+'. Версии: '+', '.join(k+' '+v for k,v in e['versions'].items())+'.')
for r in e['small_runs']+e['runs']:
    para(r['weights'])
para('Для повторения validation-части из корня проекта: MPLBACKEND=Agg .venv/bin/python reports/2026-11-01/run_experiments.py. Для сборки отчёта без обучения: MPLBACKEND=Agg .venv/bin/python reports/2026-11-01/build_report.py. final_evaluation.py выполняет test и не нужен для обычного просмотра отчёта. selected_data.npz хранит использованные срезы, source_snapshot — исходники этой серии, JSON — версии и SHA-256 кода. Скрипт обучения создаёт новые веса при каждом запуске, поэтому не запускать его лишь ради открытия отчёта.')

(OUT/'report_ru.md').write_text('\n'.join(md))
css='''body{margin:0;color:#1e293b;background:#f1f5f9;font:17px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1050px;margin:30px auto;padding:42px;background:white;border-radius:12px}h1{font-size:34px;line-height:1.2;color:#17365b}h2{font-size:25px;line-height:1.35;margin-top:40px;color:#17365b}p{overflow-wrap:anywhere}figure{margin:26px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:14px;color:#526174;margin-top:8px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border-bottom:1px solid #d9e2ef;padding:10px;text-align:left}th{background:#edf3fa}tr:nth-child(even){background:#f8fafc}@media(max-width:700px){main{padding:18px;margin:0;border-radius:0}h1{font-size:27px}h2{font-size:22px}}@media print{body{background:white;font-size:11pt}main{max-width:none;margin:0;padding:0}h2{break-after:avoid}figure,tr{break-inside:avoid}}'''
(OUT/'report_ru.html').write_text('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HIGGS — диагностика обучения</title><style>'+css+'</style></head><body><main>'+''.join(parts)+'</main></body></html>')
with (OUT/'validation_summary.csv').open('w',newline='') as file:
    fields=['model','learning_rate','training_rows','best_epoch','train_bce_at_best','validation_bce','validation_accuracy','validation_precision','validation_recall','validation_f1','validation_roc_auc','weights']
    w=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(runs)
with (OUT/'test_summary.csv').open('w',newline='') as file:
    w=csv.DictWriter(file,fieldnames=list(e['test'][0]));w.writeheader();w.writerows(e['test'])
print('Built report_ru.md, report_ru.html, seven PNG figures and two CSV tables.')
