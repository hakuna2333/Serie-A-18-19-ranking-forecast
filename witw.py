# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import math
import csv
import random
import numpy as np
import matplotlib.pyplot as mp
from sklearn import linear_model

# 数据来源 :http://www.football-data.co.uk/italym.php
# 设置回归训练时所需用到的参数变量：

# 当每支队伍没有elo等级分时，赋予其基础elo等级分
base_elo = 1600
team_elos = {}
team_stats = {}
base_score = 0
team_score = {}
X = []
y = []


# 获取每支队伍的Elo Score等级分函数，当在开始没有等级分时，将其赋予初始base_elo值：
def get_elo(team):
    try:
        return team_elos[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]


# 定义计算每支球队的Elo等级分函数：

# 计算每个球队的elo值 eq代表平局,客场球队加100分
def calc_elo(win_team, lose_team, eq=0):
    if eq == 1:
        return get_elo(win_team), get_elo(lose_team) + 100
    else:
        winner_rank = get_elo(win_team)
        loser_rank = get_elo(lose_team)

        rank_diff = winner_rank - loser_rank
        exp = (rank_diff * -1) / 400
        odds = 1 / (1 + math.pow(10, exp))
        # 根据rank级别修改K值
        if winner_rank < 2100:
            k = 32
        elif 2400 > winner_rank >= 2100:
            k = 24
        else:
            k = 16
        new_winner_rank = round(winner_rank + (k * (1 - odds)))
        new_rank_diff = new_winner_rank - winner_rank
        new_loser_rank = loser_rank - new_rank_diff

        return new_winner_rank, new_loser_rank


# 基于我们初始好的统计数据，及每支队伍的Elo score计算结果，建立对应数据集
# 在主客场比赛时，为主场作战的队伍更加有优势一点，因此会给主场作战队伍相应加上100等级分

def build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    elo_list = {'Juventus': [], 'Lazio': [], 'Fiorentina': [], 'Genoa': [], 'Inter': [],
                'Napoli': [], 'Palermo': [], 'Parma': [], 'Reggina': [], 'Siena': [],
                'Empoli': [], 'Atalanta': [], 'Cagliari': [], 'Catania': [], 'Livorno': [],
                'Roma': [], 'Sampdoria': [], 'Torino': [], 'Udinese': [], 'Milan': [],
                'Chievo': [], 'Bologna': [], 'Lecce': [], 'Bari': [], 'Cesena': [],
                'Brescia': [], 'Novara': [], 'Pescara': [], 'Verona': [], 'Sassuolo': [],
                'Frosinone': [], 'Carpi': [], 'Crotone': [], 'Benevento': [], 'Spal': []}

    for index, row in all_data.iterrows():
        # 如果是平局
        if row['FTR'] == 'D':
            Wteam = row['HomeTeam']
            Lteam = row['AwayTeam']
            # 根据这场比赛的数据更新队伍的elo值
            new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam, eq=1)
            team_elos[Wteam] = new_winner_rank
            team_elos[Lteam] = new_loser_rank

            elo_list[Wteam].append(team_elos[Wteam])
            elo_list[Lteam].append(team_elos[Lteam])

        if row['FTR'] == 'H':
            Wteam = row['HomeTeam']
            Lteam = row['AwayTeam']

            # 获取最初的elo或是每个队伍最初的elo值
            team1_elo = get_elo(Wteam)
            team2_elo = get_elo(Lteam)

            # 给主场比赛的队伍加上100的elo值
            team1_elo += 100

            # 把elo当为评价每个队伍的第一个特征值
            team1_features = [team1_elo]
            team2_features = [team2_elo]

            # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
            # 并将对应的0/1赋给y值
            if random.random() > 0.5:
                X.append(team1_features + team2_features)
                y.append(0)
            else:
                X.append(team2_features + team1_features)
                y.append(1)

            if skip == 0:
                print(X)
                skip = 1

            # 根据这场比赛的数据更新队伍的elo值
            new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
            team_elos[Wteam] = new_winner_rank
            team_elos[Lteam] = new_loser_rank

            elo_list[Wteam].append(team_elos[Wteam])
            elo_list[Lteam].append(team_elos[Lteam])

        if row['FTR'] == 'A':
            Wteam = row['AwayTeam']
            Lteam = row['HomeTeam']

            # 获取最初的elo或是每个队伍最初的elo值
            team1_elo = get_elo(Wteam)
            team2_elo = get_elo(Lteam)

            # 给主场比赛的队伍加上100的elo值
            team2_elo += 100

            # 把elo当为评价每个队伍的第一个特征值
            team1_features = [team1_elo]
            team2_features = [team2_elo]

            # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
            # 并将对应的0/1赋给y值
            if random.random() > 0.5:
                X.append(team1_features + team2_features)
                y.append(0)
            else:
                X.append(team2_features + team1_features)
                y.append(1)

            if skip == 0:
                print(X)
                skip = 1

            # 根据这场比赛的数据更新队伍的elo值
            new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
            team_elos[Wteam] = new_winner_rank
            team_elos[Lteam] = new_loser_rank

            elo_list[Wteam].append(team_elos[Wteam])
            elo_list[Lteam].append(team_elos[Lteam])

    for i in list(elo_list):
        while len(elo_list[i]) < 500:
            elo_list[i].append(elo_list[i][-1])

    x = np.linspace(0, 499, 500)
    y1 = np.array(elo_list['Juventus'])
    y2 = np.array(elo_list['Crotone'])
    y3 = np.array(elo_list['Benevento'])
    y4 = np.array(elo_list['Spal'])
    y5 = np.array(elo_list['Carpi'])
    y6 = np.array(elo_list['Frosinone'])
    y7 = np.array(elo_list['Sassuolo'])
    y8 = np.array(elo_list['Verona'])
    y9 = np.array(elo_list['Pescara'])
    y10 = np.array(elo_list['Novara'])
    y11 = np.array(elo_list['Brescia'])
    y12 = np.array(elo_list['Cesena'])
    y13 = np.array(elo_list['Bari'])
    y14 = np.array(elo_list['Lecce'])
    y15 = np.array(elo_list['Bologna'])
    y16 = np.array(elo_list['Chievo'])
    y17 = np.array(elo_list['Milan'])
    y18 = np.array(elo_list['Udinese'])
    y19 = np.array(elo_list['Torino'])
    y20 = np.array(elo_list['Sampdoria'])
    y21 = np.array(elo_list['Roma'])
    y22 = np.array(elo_list['Livorno'])
    y23 = np.array(elo_list['Catania'])
    y24 = np.array(elo_list['Cagliari'])
    y25 = np.array(elo_list['Atalanta'])
    y26 = np.array(elo_list['Empoli'])
    y27 = np.array(elo_list['Siena'])
    y28 = np.array(elo_list['Parma'])
    y29 = np.array(elo_list['Reggina'])
    y30 = np.array(elo_list['Palermo'])
    y31 = np.array(elo_list['Napoli'])
    y32 = np.array(elo_list['Inter'])
    y33 = np.array(elo_list['Genoa'])
    y34 = np.array(elo_list['Fiorentina'])
    y35 = np.array(elo_list['Lazio'])

    mp.xticks([100, 200, 300, 400, 500],
              [100, 200, 300, 400, 500])
    ax = mp.gca()
    mp.plot(x, y1, linestyle='-', linewidth=1,
            label=r'Juventus')
    mp.plot(x, y2, linestyle='-', linewidth=1,
            label=r'Crotone')
    mp.plot(x, y3, linestyle='-', linewidth=1,
            label=r'Benevento')
    mp.plot(x, y4, linestyle='-', linewidth=1,
            label=r'Spal')
    mp.plot(x, y5, linestyle='-', linewidth=1,
            label=r'Carpi')
    mp.plot(x, y6, linestyle='-', linewidth=1,
            label=r'Frosinone')
    mp.plot(x, y7, linestyle='-', linewidth=1,
            label=r'Sassuolo')
    mp.plot(x, y8, linestyle='-', linewidth=1,
            label=r'Verona')
    mp.plot(x, y9, linestyle='-', linewidth=1,
            label=r'Pescara')
    mp.plot(x, y10, linestyle='-', linewidth=1,
            label=r'Novara')
    mp.plot(x, y11, linestyle='-', linewidth=1,
            label=r'Brescia')
    mp.plot(x, y12, linestyle='-', linewidth=1,
            label=r'Cesena')
    mp.plot(x, y13, linestyle='-', linewidth=1,
            label=r'Bari')
    mp.plot(x, y14, linestyle='-', linewidth=1,
            label=r'Lecce')
    mp.plot(x, y15, linestyle='-', linewidth=1,
            label=r'Bologna')
    mp.plot(x, y16, linestyle='-', linewidth=1,
            label=r'Chievo')
    mp.plot(x, y17, linestyle='-', linewidth=1,
            label=r'Milan')
    mp.plot(x, y18, linestyle='-', linewidth=1,
            label=r'Udinese')
    mp.plot(x, y19, linestyle='-', linewidth=1,
            label=r'Torino')
    mp.plot(x, y20, linestyle='-', linewidth=1,
            label=r'Sampdoria')
    mp.plot(x, y21, linestyle='-', linewidth=1,
            label=r'Roma')
    mp.plot(x, y22, linestyle='-', linewidth=1,
            label=r'Livorno')
    mp.plot(x, y23, linestyle='-', linewidth=1,
            label=r'Catania')
    mp.plot(x, y24, linestyle='-', linewidth=1,
            label=r'Cagliari')
    mp.plot(x, y25, linestyle='-', linewidth=1,
            label=r'Atalanta')
    mp.plot(x, y26, linestyle='-', linewidth=1,
            label=r'Empoli')
    mp.plot(x, y27, linestyle='-', linewidth=1,
            label=r'Siena')
    mp.plot(x, y28, linestyle='-', linewidth=1,
            label=r'Parma')
    mp.plot(x, y29, linestyle='-', linewidth=1,
            label=r'Reggina')
    mp.plot(x, y30, linestyle='-', linewidth=1,
            label=r'Palermo')
    mp.plot(x, y31, linestyle='-', linewidth=1,
            label=r'Napoli')
    mp.plot(x, y32, linestyle='-', linewidth=1,
            label=r'Inter')
    mp.plot(x, y33, linestyle='-', linewidth=1,
            label=r'Genoa')
    mp.plot(x, y34, linestyle='-', linewidth=1,
            label=r'Fiorentina')
    mp.plot(x, y35, linestyle='-', linewidth=1,
            label=r'Lazio')
    mp.xlabel('game_num', fontsize=14)
    mp.ylabel('elo_score', fontsize=14)
    mp.legend(bbox_to_anchor=(1.01, 1), loc=2, borderpad=0, borderaxespad=0, handleheight=1, labelspacing=0,
              fontsize=6, handlelength=0.7)

    mp.rcParams['figure.dpi'] = 300
    mp.rcParams['savefig.dpi'] = 300
    mp.savefig('eloscore.png')

    return np.nan_to_num(X), y


# 利用训练好的模型在16~17年的常规赛数据中进行预测。利用模型对一场新的比赛进行胜负判断，并返回其胜利的概率：

def predict_winner(team_1, team_2, model):
    features = []
    # team 1，客场队伍
    features.append(get_elo(team_1))
    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)

    features = np.nan_to_num(features)
    return model.predict_proba([features])


# 球队赢了得三分，平局各加一分，输了不加分
def get_score(team):
    try:
        return team_score[team]
    except:
        # 当最初没有score时，给每个队伍最初赋base_score
        team_score[team] = base_score
        return team_score[team]


# 最终在main函数中调用这些数据处理函数，使用sklearn的Logistic Regression方法建立回归模型：

if __name__ == '__main__':

    result_data = pd.read_csv('./data/history.csv')

    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # 利用训练好的model在1819年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv('./data/schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['HT']
        team2 = row['VT']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            score_now = get_score(winner)
            score_new = score_now + 3
            team_score[winner] = score_new

            result.append([winner, loser, prob])

        elif prob == 0.5:
            winner = team1
            loser = team2
            score_1 = get_score(winner) + 1
            score_2 = get_score(loser) + 1
            team_score[winner] = score_1
            team_score[loser] = score_2

            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            score_now = get_score(winner)
            score_new = score_now + 3
            team_score[winner] = score_new

            result.append([winner, loser, 1 - prob])
    # 计算最终排名
    ranking = []
    for item in team_score:
        ranking.append([item, team_score[item]])
    # 对分数进行排序
    ranking = sorted(ranking, reverse=True, key=lambda x: x[1])

    # 将预测结果输出到1819Result.csv文件中, 输出排名结果到1819Ranking.csv
    # 输出赢球概率文件
    with open('1819Result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['winner', 'loser', 'probability'])
        writer.writerows(result)
    # 输出排名文件
    with open('1819Ranking.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['team', 'score'])
        writer.writerows(ranking)

    l = []
    m = []
    for i in ranking:
        m.append(i[0])
        l.append(i[1])
    n = len(ranking)
    x = np.arange(n)
    y = np.array(l)
    mp.figure('Bar')
    mp.title('18-19 SERIE A ranking forecast', fontsize=20)
    mp.xlabel('team', fontsize=14)
    mp.ylabel('score', fontsize=14)
    mp.xticks(x, m)
    mp.tick_params(rotation=30, labelsize=8)
    mp.bar(x, y, ec='white')
    for _x, _y in zip(x, y):
        mp.text(_x, _y + 3, '%.0f' % _y,
                ha='center', va='top', size=8)
    mp.savefig('ranking.png')
