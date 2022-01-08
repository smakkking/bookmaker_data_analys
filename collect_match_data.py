import csv
import pickle
from requests.sessions import session
import steam.webauth as wa
from bs4 import BeautifulSoup as bs
from time import sleep
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from itertools import product

CF_CLEARANCE = 'ALNG.TfELa3npJvsbG1ur_NKO1ZOEwWNYwalquRnDK8-1641667303-0-150'
MOZILLA_UA = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'

def calculate_kef(k1, k2):
    u_fee_reduce = 48
    fee_kef = 1 - (10 - u_fee_reduce * 1 / 100) / 100
    return round((k1*fee_kef/k2+1), 2)


def login_steam():
    client = {'bookmaker_login': 'ft1w1mcx1', 'bookmaker_password': 'nyzpvnsd1'}
    user = wa.WebAuth(client['bookmaker_login'])
    session = user.login(client['bookmaker_password'])

    head = {'User-Agent': MOZILLA_UA}
    session.headers.update(head)
    session.cookies.set(
        'cf_clearance', 
        CF_CLEARANCE, 
    )

    r = session.get('https://betscsgo.vip/login/')

    sleep(1)

    soup = bs(r.text, 'lxml')
    form_obj = soup.find(id='openidForm')

    r = session.post('https://steamcommunity.com/openid/login', files={
        'action': (None, form_obj.find('input', {'id': 'actionInput'})['value']),
        'openid.mode': (None, form_obj.find('input', {'name': 'openid.mode'})['value']),
        'openidparams': (None, form_obj.find('input', {'name': 'openidparams'})['value']),
        'nonce': (None, form_obj.find('input', {'name': 'nonce'})['value'])
    })

    sleep(2)

    with open('match_data_session', 'wb') as f:
        pickle.dump(session, f)


def machine_learning_predict():
    df = pd.read_csv('match_data.csv')

    print('Размер выборки:', df.shape[0])

    train, test = train_test_split(df, test_size=0.1, random_state=0)
    logreg = LogisticRegression()

    x_train = list(zip(train['left_sum'], train['right_sum']))
    x_test = list(zip(test['left_sum'], test['right_sum']))

    logreg.fit(x_train, train['min_win'])

    # hus in binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, test['min_win'])))

    confusion = confusion_matrix(test['min_win'], y_pred)
    print(confusion)

    logit_roc_auc = roc_auc_score(test['min_win'], y_pred)
    fpr, tpr, thresholds = roc_curve(test['min_win'], logreg.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def load_data(data_size, login=False):
    if login:
        login_steam()

    with open('match_data_session', 'rb') as f:
        session = pickle.load(f)

    with open('match_data.csv', 'w') as f:
        Writer = csv.writer(f)
        Writer.writerow(['left_sum', 'right_sum', 'kef1', 'kef2', 'min_win'])
    
        dataframe = []

        for i in range(1, data_size + 1):
            print(f'Page{i} processed ...')

            t = session.get(f'https://betscsgo.vip/history/{i}/').text
            sleep(2.2)

            beg_pos = t.find('var bets =') + len('var bets =')
            end_pos = t.find("""$(function () {
                    if (bets.length == 0)""")

            t = t[beg_pos : end_pos]
            t = t[ : t.find(';')]

            t = json.loads(t)
                
            for match in t:
                if match['m_bets_a'] * match['m_bets_b'] == 0 :
                    continue

                dataframe.append(
                    [
                        match['m_bets_a'] / 100,
                        match['m_bets_b'] / 100,
                        calculate_kef(match['m_bets_b'], match['m_bets_a']),
                        calculate_kef(match['m_bets_a'], match['m_bets_b']),
                        (match['m_status'] == '2' and match['m_bets_a'] > match['m_bets_b']) or (match['m_status'] == '3' and match['m_bets_a'] < match['m_bets_b']),
                    ]
                )

            if len(dataframe) > 100:
                Writer.writerows(dataframe)
                dataframe = []

        Writer.writerows(dataframe)

    print('Ended')

def stat_pred_max_kef():
    df = pd.read_csv('match_data.csv')
    df['max_kef'] = pd.Series(list(map(max, zip(df['kef1'], df['kef2']))))
    df['max_win'] = df['min_win'].apply(lambda x: not x)

    best = {
        'R_max': 0,
        'a': 1.0,
        'b': 20,
        'p': 0,
        'k': 0
    }
    def filter(df, a, b):
        return df[(df['max_kef'] > a) & (df['max_kef'] < b)]
    
    N_DEL = 50
    for (a, b) in product(list(np.linspace(1.1, 20, N_DEL)), list(np.linspace(1.1, 20, N_DEL))) :
        if a >= b :
            continue
        df1 = filter(df, a, b)

        number_of_min_win = df1['max_win'].apply(int)
        
        if not (True in number_of_min_win):
            continue
        # частотная оценка вероятности
        p = sum(number_of_min_win) / len(number_of_min_win)
        # средний к-т
        k = df1[df1['max_win']]['max_kef'].mean()

        R = k*p-1

        if R > best['R_max']:
            best['R_max'] = R
            best['a'], best['b'] = a, b
            best['p'], best['k'] = p, k

    print(f"Чистая доходность={int(best['R_max'] * 100)}")
    print(f"Достигнута на a={best['a']}, b={best['b']}")
    if best['R_max'] > 0:
        print(f"Доходность на выходе={((1+best['R_max']) * 0.95 - 1)*100}")
        print(f"частотная оценка вероятности={best['p']}, средний к-т={best['k']}")

    print(f"Стата для макс к-та")
    result_sum = 0
    for i in range(df.shape[0]):
        result_sum += df['max_kef'][i] if df['max_win'].iloc[i] else -1

    print(f"реальная доходность={(result_sum * 0.95 / df.shape[0] - 1)}\n")


def stat_pred_min_kef():       
    df = pd.read_csv('match_data.csv')
    df['min_kef'] = pd.Series(list(map(min, zip(df['kef1'], df['kef2']))))

    def filter(df, a=1.0, b=2.0):
        return df[(df['min_kef'] > a) & (df['min_kef'] < b)]

    best = {
        'R_max': 0,
        'a': 1.0,
        'b': 2.0
    }
    N_DEL = 10
    for (a, b) in product(list(np.linspace(1.1, 2.0, N_DEL)), list(np.linspace(1.1, 2.0, N_DEL))) :
        if a >= b :
            continue
        df1 = filter(df, a, b)

        number_of_min_win = df1['min_win'].apply(int)
        
        if not (True in number_of_min_win):
            continue
        # частотная оценка вероятности
        p = sum(number_of_min_win) / len(number_of_min_win)
        # средний к-т
        k = df1[df1['min_win']]['min_kef'].mean()

        R = k*p-1

        if R > best['R_max']:
            best['R_max'] = R
            best['a'], best['b'] = a, b

    print(f"Чистая доходность={int(best['R_max'] * 100)}")
    print(f"Достигнута на a={best['a']}, b={best['b']}")
    if best['R_max'] > 0:
        print(f"Доходность на выходе={((1+best['R_max']) * 0.95 - 1)*100}")

    print(f"Стата для мин к-та")
    result_sum = 0
    for i in range(df.shape[0]):
        result_sum += df['min_kef'][i] if df['min_win'].iloc[i] else -1

    print(f"реальная доходность={(result_sum * 0.95 / df.shape[0] - 1)}")

stat_pred_min_kef()
