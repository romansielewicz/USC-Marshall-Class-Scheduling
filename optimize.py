# %% markdown
# # Gurobi Formulation Round 2
# %% codecell
import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB, max_, min_
# %% codecell
# Defining department to optimize
# %% codecell
# Defining to_minutes
# Converts Strings of time to minutes since midnight


def to_minutes(x):
    pos = x.index(':')
    hours = x[:pos]
    minutes = x[pos+1:pos+3]
    hours = int(hours)
    minutes = int(minutes)
    total = 60*hours + minutes
    return total

# %% codecell


def depOptimizer(base_schedule, allocations, dep):
    df = base_schedule[base_schedule['department'] == dep].copy()
    #df = df[df['level'] == 'UG']
    #df = df[df['first_days'].isin(['MW', 'TH'])]
    df.index = df['section']
    # Defining J
    J = list(allocations.columns)
    # Defining F
    F = pd.Series(allocations.iloc[0, :], dtype='float64').copy()
    # Defining A
    A = pd.Series(index=J, dtype='float64')
    for j in J:
        val = allocations.loc[allocations[j] == dep, j].count()
        A[j] = 4 * val
    # Defining I
    I = list(df.index)
    # Defining S
    S = pd.Series(df.loc[:, 'seats_offered'], dtype='float64').copy()
    # Defining L
    L = pd.Series(df.loc[:, 'periods'], dtype='float64').copy()
    # Gurobi Optimization
    mod = Model()
    X = mod.addVars(I, J, vtype=GRB.BINARY)
    mod.setObjective(sum(F[j]*X[i, j] - S[i]*X[i, j] for i in I for j in J))
    # i
    for j in J:
        mod.addConstr(sum(L[i]*X[i, j] for i in I) <= A[j])
    # ii
    for i in I:
        for j in J:
            mod.addConstr(S[i]*X[i, j] <= F[j])
    # iii
    for i in I:
        mod.addConstr(sum(X[i, j] for j in J) == 1)
    mod.setParam('outputflag', False)
    mod.optimize()
    # Creating Decision vars dataframe
    output = pd.DataFrame(index=I,
                          columns=['Class_Section', 'Room',
                                   'Course', 'Students', 'Capacity'])
    output['Class_Section'] = output.index
    for i in I:
        for j in J:
            if X[i, j].x:
                room = j
        output.loc[i, 'Room'] = room
        output.loc[i, 'Course'] = df.loc[i, 'course']
        output.loc[i, 'Students'] = S[i]
        output.loc[i, 'Capacity'] = F[output.loc[i, 'Room']]
    output['Average Empty Seats per Class'] = np.nan
    output = output.reset_index(drop=True)
    output.loc[0, 'Average Empty Seats per Class'] = round(mod.objVal/len(I), 2)
    return(output)


# %% codecell
# Testing depOptimizer
# inputFile = './data/input.xlsx'
# base_schedule = pd.read_excel(inputFile,
#                               sheet_name='base_schedule')
# allocations = pd.read_excel(inputFile,
#                             sheet_name='UG_Master_Distribute')
# # Removing TBA
# base_schedule = base_schedule.loc[base_schedule['first_begin_time'] != 'TBA', :].copy()
# # converting date time to string
# base_schedule['first_begin_time'] = base_schedule['first_begin_time'].apply(str)
# base_schedule['first_end_time'] = base_schedule['first_end_time'].apply(str)
# # Defining periods: The amount of half hour blocks a class occupies
# base_schedule['begin_time'] = base_schedule['first_begin_time'].apply(lambda x: to_minutes(x))
# base_schedule['end_time'] = base_schedule['first_end_time'].apply(lambda x: to_minutes(x))
# base_schedule['length'] = base_schedule['end_time'] - base_schedule['begin_time']
# base_schedule['periods'] = np.ceil(base_schedule['length']/30)
# base_schedule = base_schedule.loc[base_schedule['level'] == 'UG'].copy()
# allocations = allocations.iloc[:, 1:].copy()

# %% codecell
# D = list(base_schedule['department'].unique())
# # for dep in D:
# #     print(dep)
# #base_schedule[base_schedule['department'] == 'GSBA']
# D = [
#     # 'ACCT',
#     'BAEP',
#     # 'BUAD',
#     # 'FBE',
#     'BUCO',
#     # 'MOR',
#     # 'MKT',
#     'DSO'
# ]

# %% codecell
# Testing depOptimizer
# for dep in D:
#     print(dep)
#     print(depOptimizer(base_schedule, allocations, dep))

# %% codecell


def optimize(inputFile, outputFile):
    base_schedule = pd.read_excel(inputFile,
                                  sheet_name='base_schedule')
    allocations = pd.read_excel(inputFile,
                                sheet_name='UG_Master_Distribute')
    # Removing TBA
    base_schedule = base_schedule.loc[base_schedule['first_begin_time'] != 'TBA', :].copy()
    # converting date time to string
    base_schedule['first_begin_time'] = base_schedule['first_begin_time'].apply(str)
    base_schedule['first_end_time'] = base_schedule['first_end_time'].apply(str)
    # Defining periods: The amount of half hour blocks a class occupies
    base_schedule['begin_time'] = base_schedule['first_begin_time'].apply(lambda x: to_minutes(x))
    base_schedule['end_time'] = base_schedule['first_end_time'].apply(lambda x: to_minutes(x))
    base_schedule['length'] = base_schedule['end_time'] - base_schedule['begin_time']
    base_schedule['periods'] = np.ceil(base_schedule['length']/30)
    # subsetting for Undergrad courses
    base_schedule = base_schedule.loc[base_schedule['level'] == 'UG'].copy()
    # subsetting for MW and TH
    base_schedule = base_schedule[base_schedule['first_days'].isin(['MW', 'TH', 'MWF'])]
    # Cutting first columns from allocations
    allocations = allocations.iloc[:, 1:].copy()
    #D = ['BAEP', 'BUCO', 'DSO']
    # Defining list of departments
    D = list(base_schedule['department'].unique())
    assignments = {}
    # Defining a different df for each department
    for dep in D:
        assignments[dep] = depOptimizer(base_schedule, allocations, dep)
    # Saving to Excel
    with pd.ExcelWriter(outputFile) as writer:
        for dep in D:
            output = assignments[dep]
            output.to_excel(writer, sheet_name=dep + ' Class Allocations', index=False)


# %% codecell
# optimize('./data/input.xlsx', './data/output.xlsx')

# %% codecell
if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 3:
        print('Correct syntax: python optimize.py inputFile outputFile')
    else:
        inputFile = sys.argv[1]
        outputFile = sys.argv[2]
    if os.path.exists(inputFile):
        optimize(inputFile, outputFile)
        print(f'Successfully optimized. Results in "{outputFile}"')
    else:
        print(f'File "{inputFile}" not found!')
