import argparse
import random
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MAX_CONF = 5

def anonymize(filename):
    xls = pd.ExcelFile(filename)
    df1 = pd.read_excel(xls, 'Relecture initiales')
    df2 = pd.read_excel(xls, 'PC')
    df3 = pd.read_excel(xls, 'Rang notes')

    # 'Relevance', 'Overall', 'Originality', 'Related', 'Significance', 'Quality', 'Method'
    df1 = df1[['#Paper', 'PC member', 'Total', 'Confidence']]
    df1['Submission'] = df1['#Paper'].apply(lambda x: int(x[1:]))

    # PCMember, RefRelecteur, PrénomRelecteur, NomRelecteur
    df2['PC member'] = df2.apply(lambda r: r['First name'] + ' ' + r['Last name'], axis=1)
    # df2['RefRelecteur'] = df2['RefRelecteur'].apply(lambda x: int(x[2:]))
    df2['RefID'] = df2.index
    df2 = df2[['RefID', 'PC member']]

    df3 = df3[['#', 'Decision']]

    df = pd.merge(df1, df3, left_on='Submission', right_on='#')
    del df['#']
    res = pd.merge(df, df2, on='PC member')
    #res = res[['Submission', 'RefRelecteur', 'Decision', 'Total', 'Confidence']]
    res = res.sort_values('Submission')
    return res

def preprocess(df):
    # df = df[['Submission', 'RefRelecteur', 'Total', 'Confidence', 'Decision']]
    df['Submission'] = df.apply(lambda x: np.where(submissions == x['Submission'])[0][0], axis=1)
    df['RefID'] = df.apply(lambda x: np.where(reviewers == x['RefID'])[0][0], axis=1)
    df['Decision'] = df.apply(lambda x: x['Decision'].startswith('ACCEPT') if pd.notnull(x['Decision']) else x['Decision'], axis=1)
    df = df.sort_values('RefID')
    df['Score'] = df.apply(lambda r: r['Total'], axis=1) #  + r['Relevance']
    df['Confidence'] = df.apply(lambda r: r['Confidence'] / MAX_CONF, axis=1)
    return df[['Submission', 'RefID', 'Score', 'Confidence', 'Decision']]

def OPCA(df, q, w, level=0):
    submissions = df.Submission.unique()
    submissions = np.sort(submissions)
    points = np.zeros((len(submissions), len(submissions), 2))
    scores = np.zeros(len(submissions))
    results = []
    
    grouped = df.groupby('RefID')
    for _, group in grouped:
        group = group.sort_values('Score')
        for index, (_, rowA) in enumerate(group.iterrows()):
            for _, rowB in list(group.iterrows())[index + 1:]:
                i = np.where(submissions == rowA['Submission'])[0][0]
                j = np.where(submissions == rowB['Submission'])[0][0]
                points[i][j][1] += 1
                points[j][i][1] += 1
                if level == 0:
                    if rowA['Score'] == rowB['Score']:
                        points[i][j][0] += 0.5
                        points[j][i][0] += 0.5
                    else:
                        points[j][i][0] += 1
                else:
                    points[j][i][0] += (rowB['Score'] - rowA['Score']) * rowA['Confidence']

    # print(points[:,:,1])

    for i in range(len(submissions)):
        points[i][i][0] = None
        for j in range(i + 1, len(submissions)):
            m = points[i][j][1]
            if m < q:
                points[i][j][0] = points[j][i][0] = None
            else:
                if points[i][j][0] == points[j][i][0]:
                    points[i][j][0] = points[j][i][0] = 0.5
                elif points[i][j][0] > points[j][i][0]:
                    points[i][j][0] = 1
                    points[j][i][0] = 0
                else:
                    points[i][j][0] = 0
                    points[j][i][0] = 1

    #print(points[:,:,0])
    
    if level > 1:
        for i in range(len(submissions)):
            for j in range(i + 1, len(submissions)):
                if points[i][j][1] == 0:
                    xi = xj = 0
                    for k in range(len(submissions)):
                        if k == i or k == j:
                            continue
                        if points[i][k][0] > points[j][k][0]:
                            xi += 1
                        elif points[i][k][0] < points[j][k][0]:
                            xj += 1
                    #print(i, j, xi, xj, xi + xj)
                    if xi == xj:
                        points[i][j][0] = points[j][i] = w * 0.5
                    elif xi > xj:
                        points[i][j][0] = w
                        points[j][i][0] = 0
                    else:
                        points[i][j][0] = w
                        points[j][i][0] = 0
        #print(points[:,:,0])
    for i in range(len(submissions)):
        lst = [x[0] for x in points[i] if not np.isnan(x[0])]
        if len(lst) == 0:
            scores[i] = 0
        else:
            scores[i] = np.mean(np.array(lst))
    inds = np.argsort(scores)
    for ind in reversed(inds):
        results.append((submissions[ind], scores[ind]))

    return np.array(results)

def standardization(df):
    ref_grp = df.groupby(['RefID'])
    ref_grp = ref_grp['Score'].agg([np.mean, np.std])
    ref_grp['RefID'] = ref_grp.index
    ref_grp.reset_index(drop=True, inplace=True)
    df = pd.merge(df, ref_grp)
    df = df.assign(Score = lambda x: (x['Score'] - x['mean']) / x['std'])
    del df['mean']
    del df['std']
    return df

def average(df, standard=False):
    if standard:
        df = standardization(df)
    grouped = df.groupby(['Submission']).mean()
    scores = np.zeros(len(grouped))
    for x in grouped.itertuples():
        scores[x.Index] = x.Score
    inds = np.argsort(scores)
    results = []
    for ind in reversed(inds):
        results.append((ind, scores[ind]))
    return np.array(results)

def brute_force(df):
    pass

def check(ranking):
    pass

def modify_edge(i, j, cntP, cntR, edges, action):
    cntR[i] += action - edges[i][j]
    cntP[j] += action - edges[i][j]
    edges[i][j] = action

def build_data(P, R, p, r):
    edges = np.zeros((R, P))
    cntR = np.zeros(R)
    cntP = np.zeros(P)

    for i in range(R):
        submissions = random.sample(range(P), p)
        for j in submissions:
            modify_edge(i, j, cntP, cntR, edges, 1)

    for j in range(P):
        reviewers = random.sample(range(R), r)
        for i in reviewers:
            modify_edge(i, j, cntP, cntR, edges, 1)

    data = []
    for j in range(P):
        perm = np.random.permutation(R)
        for i in perm:
            if cntR[i] > p and cntP[j] > r:
                modify_edge(i, j, cntP, cntR, edges, 0)
            if edges[i][j] == 1:
                data.append([i, j, random.randint(0, 20), 5, None])
    data = np.array(data)
    return pd.DataFrame({'Submission': data[:, 1], 'RefID': data[:, 0], 'Total': data[:, 2], 'Confidence': data[:, 3], 'Decision': data[:, 4]})

def conflicts(df, ranking):
    duels = []
    trans = []
    decision = []
    grouped = df.groupby('RefID')
    submissions = df.Submission.unique()
    ranking = np.array([list(x) for x in ranking])

    for _, group1 in grouped:
        group1 = group1.sort_values('Score')
        ref1 = list(group1.RefID)[0]
        for _, group2 in grouped:
            group2 = group2.sort_values('Score')
            ref2 = list(group2.RefID)[0]
            if ref1 == ref2:
                continue
            for index1, (_, rowA1) in enumerate(group1.iterrows()):
                for _, rowB1 in list(group1.iterrows())[index1 + 1:]:
                    i1 = np.where(submissions == rowA1['Submission'])[0][0]
                    j1 = np.where(submissions == rowB1['Submission'])[0][0]
                    if rowA1['Score'] != rowB1['Score']:
                        for index2, (_, rowA2) in enumerate(group2.iterrows()):
                            for _, rowB2 in list(group2.iterrows())[index2 + 1:]:
                                i2 = np.where(submissions == rowA2['Submission'])[0][0]
                                j2 = np.where(submissions == rowB2['Submission'])[0][0]
                                if j1 != i2:
                                    continue
                                if rowA2['Score'] != rowB2['Score']:
                                    x = np.where(ranking[:, 0] == rowA1['Submission'])[0][0]
                                    y = np.where(ranking[:, 0] == rowB2['Submission'])[0][0]
                                    if ranking[x][1] > ranking[y][1]:
                                        trans.append((x, y, ref1, ref2))
        for index1, (_, rowA1) in enumerate(group1.iterrows()):
            for _, rowB1 in list(group1.iterrows())[index1 + 1:]:
                i1 = np.where(submissions == rowA1['Submission'])[0][0]
                j1 = np.where(submissions == rowB1['Submission'])[0][0]
                if rowA1['Score'] != rowB1['Score']:
                    x = np.where(ranking[:, 0] == rowA1['Submission'])[0][0]
                    y = np.where(ranking[:, 0] == rowB1['Submission'])[0][0]
                    if ranking[x][1] > ranking[y][1]:
                        #print(x, y, ref1, ranking[x][1], ranking[y][1], rowA1['Score'], rowB1['Score'])
                        duels.append((x, y, ref1))
    duels, trans = set(duels), set(trans)
    grouped = df.groupby('Submission')
    decisions = {}
    for _, group in grouped:
        decisions[group.Submission.unique()[0]] = group.Decision.unique()[0]
    
    for i in range(ranking.shape[0]):
        if decisions[ranking[i][0]]:
            continue
        for j in range(i + 1, ranking.shape[0]):
            if decisions[ranking[j][0]]:
                decision.append((ranking[i][0], ranking[j][0]))

    return duels, trans, decision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-f', dest='filename', type=str, help='Input filename')
    parser.add_argument('-R', dest='R', type=int, help='Number of Reviewers')
    parser.add_argument('-P', dest='P', type=int, help='Number of Submissions')
    parser.add_argument('-r', dest='r', type=int, help='Minimum count of reviews for each paper')
    parser.add_argument('-p', dest='p', type=int, help='Minimum count of submissions to review for each reviewer')
    parser.add_argument('-q', dest='q', type=int, help='OPCA q threshold', default=1)
    parser.add_argument('-w', dest='w', type=int, help='Modified OPCA w weight', default=1)

    args = parser.parse_args()

    P, R = args.P, args.R
    p, r = args.p, args.r
    q = args.q
    w = args.w
    
    if args.filename:
        if args.filename.endswith('csv'):
            df = pd.read_csv(args.filename)
        else:
            df = anonymize(args.filename)
    else:
        df = build_data(P, R, p, r)

    submissions = df.Submission.unique()
    reviewers = df.RefID.unique()
    N = len(submissions)
    M = len(reviewers)

    df = preprocess(df)

    avg = average(df, False)
    sd_avg = average(df, True)
    opca = OPCA(df, q, w, 0)
    ours = OPCA(df, q, w, 2)
    ranks = pd.DataFrame({
        'Average': list(avg), 
        'Standardization Average': list(sd_avg), 
        'Original OPCA': list(opca),
        'Our OPCA': list(ours)})

    for column in ranks:
        duels, trans, decision = conflicts(df, ranks[column])
        print(column, len(duels), len(trans), len(decision))
    
    # ranks = ranks.applymap(lambda x: [submissions[int(x[0])], x[1]])
    # print(df)
    # print(ranks)

    labels = {}
    acc_rate = 0.5
    for column in ranks:
        labels[column] = np.zeros(N, dtype=bool)
        for index, paper in enumerate(ranks[column]):
            if index >= N * acc_rate:
                break
            labels[column][int(paper[0])] = True

    # for i, x in enumerate(ranks):
    #     for j, y in enumerate(ranks):
    #         if i >= j:
    #             continue
    #         cf_matrix = confusion_matrix(labels[x], labels[y])
    #         sns.heatmap(data=cf_matrix, fmt='.0f', xticklabels=np.unique(labels['Average']), yticklabels=np.unique(labels['Average']), annot=True)
    #         plt.xlabel(x)
    #         plt.ylabel(y)
    #         plt.show()

