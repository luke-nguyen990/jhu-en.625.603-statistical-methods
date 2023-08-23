import numpy as np
import statsmodels.api as sm
import pandas as pd


class CurrentPopulationSurveyDataFrame:
    df = None

    def __init__(self):
        file_path = 'CPS2015-1.xlsx'
        file_sheet_name = 'Data'
        self.df = pd.read_excel(file_path, sheet_name=file_sheet_name)


def part_a():
    df = CurrentPopulationSurveyDataFrame().df
    Y = df['ahe']
    X = df[['age', 'female', 'bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_b():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    Y = df['ln_ahe']
    X = df[['age', 'female', 'bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_c():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    df['ln_age'] = np.log(df['age'])
    Y = df['ln_ahe']
    X = df[['ln_age', 'female', 'bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_d():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    df['age_squared'] = np.square(df['age'])
    Y = df['ln_ahe']
    X = df[['age', 'age_squared', 'female', 'bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_e():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    df['age_squared'] = np.square(df['age'])
    df['female_x_bachelor'] = df['female'] * df['bachelor']
    Y = df['ln_ahe']
    X = df[['age', 'age_squared', 'female', 'bachelor', 'female_x_bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_i():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    df['age_squared'] = np.square(df['age'])
    df['age_x_female'] = df['age'] * df['female']
    Y = df['ln_ahe']
    X = df[['age', 'age_squared', 'bachelor', 'age_x_female']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


def part_j():
    df = CurrentPopulationSurveyDataFrame().df
    df['ln_ahe'] = np.log(df['ahe'])
    df['age_squared'] = np.square(df['age'])
    df['age_x_bachelor'] = df['age'] * df['bachelor']
    Y = df['ln_ahe']
    X = df[['age', 'age_squared', 'female', 'age_x_bachelor']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print()
    print(model.summary())


if __name__ == '__main__':
    part_j()
