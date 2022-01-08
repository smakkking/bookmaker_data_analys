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


def calculate_kef(k1, k2):
    u_fee_reduce = 48
    fee_kef = 1 - (10 - u_fee_reduce * 1 / 100) / 100
    return round((k1*fee_kef/k2+1), 2)

def login_steam():
    client = {'bookmaker_login': 'ft1w1mcx1', 'bookmaker_password': 'nyzpvnsd1'}
    user = wa.WebAuth(client['bookmaker_login'])
    session = user.login(client['bookmaker_password'])

    head = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'}
    session.headers.update(head)
    session.cookies.set(
        'cf_clearance', 
        'YrvWlFVqhxBrkRYmC_BEkfOWr6nFW8ay_R_W5tocp7o-1641653059-0-150', 
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

    with open('match_data_session', 'w') as f:
        pickle.dump(session, f)


def main(data_size, login=False):
    if login:
        login_steam()

    with open('match_data_session', 'r') as f:
        session = pickle.load(f)

    dataframe = []

    for i in range(1, data_size + 1):
        print(f'Page{i} processed ...')

        t = session.get(f'https://betscsgo.vip/history/{i}/').text
        sleep(2)

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
                {
                    'left_sum': match['m_bets_a'] / 100,
                    'right_sum': match['m_bets_b'] / 100,
                    'kef1': calculate_kef(match['m_bets_b'], match['m_bets_a']),
                    'kef2': calculate_kef(match['m_bets_a'], match['m_bets_b']),
                    'min_win': (match['m_status'] == '2' and match['m_bets_a'] > match['m_bets_b']) or (match['m_status'] == '3' and match['m_bets_a'] < match['m_bets_b']),
                }
            )
    
    print('Ended')
    return dataframe

def machine_learning_predict():
    with open('match_data.json', 'r') as f:
        df = pd.DataFrame(json.load(f))

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
    with open('match_data.json', 'w') as f:
        json.dump(main(data_size, login), f, indent=4)

def statistic_predict():
    with open('match_data.json', 'r') as f:
        df = pd.DataFrame(json.load(f))
        df['min_kef'] = pd.Series(list(map(min, zip(df['kef1'], df['kef2']))))

    def filter(df, a=1.0, b=2.0):
        return df[(df['min_kef'] > a) & (df['min_kef'] < b)]

    df = filter(df, a=1.7)

    number_of_min_win = df['min_win'].value_counts()
    p = number_of_min_win.iloc[0] / (number_of_min_win.iloc[0] + number_of_min_win.iloc[1])
    k = df['min_kef'].mean()

    print(f'Частотная оценка вероятности={p}\nСредний к-т={k}')
    R = k*p-1
    print(f'Чистая доходность={int(R * 100)}')
    if R > 0:
        print(f'Доходность на выходе={(1+R) * 0,95}')


load_data(1, login=True)
#machine_learning_predict()
statistic_predict()
