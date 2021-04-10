
import pandas as pd
import numpy as np

train = pd.read_csv('../input/sepredict/태양광발전량 예측/train/train.csv')
submission = pd.read_csv('../input/sepredict/태양광발전량 예측/sample_submission.csv')
submission.set_index('id',inplace=True)


def transform(dataset, target, start_index, end_index, history_size,
                      target_size, step):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, 48):
        indices = range(i-history_size, i, step)
        data.append(np.ravel(dataset[indices].T))
        labels.append(target[i:i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

#x_col =['DHI','DNI','WS','RH','T','TARGET']
x_col =['TARGET']
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)

past_history = 48 * 2
future_target = 48 * 2

### transform train
train_data, train_label = transform(dataset, label, 0, None, past_history, future_target, 1)
### transform test
test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'../input/sepredict/태양광발전량 예측/test/{i}.csv')
    tmp = tmp.loc[:, x_col].values
    tmp = tmp[-past_history:,:]
    data.append(np.ravel(tmp.T))
    data = np.array(data)
    test.append(data)
test = np.concatenate(test, axis=0)

from sklearn import ensemble
#n_estimators=1400, max_depth=8
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [1300, 1400, 1500],
           'max_depth' : [8, 9, 10, 11],
            }

rf = ensemble.RandomForestRegressor(max_features=1, random_state=0,
                                     verbose=True,
                                     n_jobs=-1)
grid_cv = GridSearchCV(rf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(train_data, train_label)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

submission.to_csv(f'submission.csv')
