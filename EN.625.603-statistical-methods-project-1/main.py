import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import scipy.stats
import matplotlib.pyplot as plt


class LeadMortalityDataframe:

    df = None

    column_name_year = None
    column_name_city = None
    column_name_state = None
    column_name_age = None
    column_name_hardness = None
    column_name_ph = None
    column_name_infrate = None
    column_name_typhoid_rate = None
    column_name_np_tub_rate = None
    column_name_mom_rate = None
    column_name_population = None
    column_name_precipitation = None
    column_name_temperature = None
    column_name_lead = None
    column_name_foreign_share = None

    def __init__(self):
        file_path = 'lead_mortality.xlsx'
        file_sheet_name = 'Data'

        self.df = pd.read_excel(file_path, sheet_name=file_sheet_name)
        self.column_name_year = self.df.columns[0]
        self.column_name_city = self.df.columns[1]
        self.column_name_state = self.df.columns[2]
        self.column_name_age = self.df.columns[3]
        self.column_name_hardness = self.df.columns[4]
        self.column_name_ph = self.df.columns[5]
        self.column_name_infrate = self.df.columns[6]
        self.column_name_typhoid_rate = self.df.columns[7]
        self.column_name_np_tub_rate = self.df.columns[8]
        self.column_name_mom_rate = self.df.columns[9]
        self.column_name_population = self.df.columns[10]
        self.column_name_precipitation = self.df.columns[11]
        self.column_name_temperature = self.df.columns[12]
        self.column_name_lead = self.df.columns[13]
        self.column_name_foreign_share = self.df.columns[14]

    def log_dataframe_info(self):
        print(f'Number of rows: {self.df.shape[0]}')
        print(f'Number of columns: {self.df.shape[1]}')
        print('Column names:')
        print(self.df.columns.tolist())
        print('Data summary:')
        print(self.df.describe(include='all'))

    def get_lead_by_condition(self, condition):
        return self.df[self.df[self.column_name_lead] == condition]

    def get_infrate_by_lead_condition(self, lead_condition):
        df = self.get_lead_by_condition(lead_condition)
        return df[self.column_name_infrate]


def part_a_solution():
    lead_mortality = LeadMortalityDataframe()

    infrate_lead_0 = lead_mortality.get_infrate_by_lead_condition(0)
    infrate_lead_1 = lead_mortality.get_infrate_by_lead_condition(1)

    n = infrate_lead_0.shape[0]
    m = infrate_lead_1.shape[0]
    avg_x = infrate_lead_0.mean()
    avg_y = infrate_lead_1.mean()
    std_x = infrate_lead_0.std()
    std_y = infrate_lead_1.std()
    level_of_significance = 0.05
    d_freedom = n + m - 2
    t_a_df = -scipy.stats.t.ppf(level_of_significance, d_freedom)
    std_pooled = np.sqrt(
        ((n - 1) * pow(std_x, 2) + (m - 1) * pow(std_y, 2)) / d_freedom)
    t_statistics = (avg_x - avg_y) / (std_pooled * np.sqrt(1/n + 1/m))
    print(f'n: {n}')
    print(f'm: {m}')
    print(f'avg_x: {avg_x:.4f}')
    print(f'avg_y: {avg_y:.4f}')
    print(f'std_x: {std_x:.4f}')
    print(f'std_y: {std_y:.4f}')
    print(f'std_pooled: {std_pooled:.4f}')
    print(f't_a_df: {t_a_df:.4f}')
    print(f't_statistics: {t_statistics:.4f}')


def part_b_i_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]

    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph, "lead-pH"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())


def part_b_ii_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]

    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph, "lead-pH"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    model = sm.OLS(y, x)
    results = model.fit()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    lead_mortality.df['pred_infrate'] = results.predict(x)

    df_lead0 = lead_mortality.df[lead_mortality.df['lead'] == 0]
    ax[0].scatter(df_lead0['ph'], df_lead0['infrate'],
                  color='blue', alpha=0.5, label='Actual')
    sorted_df_lead0 = df_lead0.sort_values(by='ph')
    ax[0].plot(sorted_df_lead0['ph'], sorted_df_lead0['pred_infrate'],
               color='red', alpha=0.5, label='Predicted')
    ax[0].set_title('Infant Mortality Rate vs. pH (Lead = 0)')
    ax[0].set_xlabel('pH')
    ax[0].set_ylabel('Infant Mortality Rate')
    ax[0].legend()

    df_lead1 = lead_mortality.df[lead_mortality.df['lead'] == 1]
    ax[1].scatter(df_lead1['ph'], df_lead1['infrate'],
                  color='blue', alpha=0.5, label='Actual')
    sorted_df_lead1 = df_lead1.sort_values(by='ph')
    ax[1].plot(sorted_df_lead1['ph'], sorted_df_lead1['pred_infrate'],
               color='red', alpha=0.5, label='Predicted')
    ax[1].set_title('Infant Mortality Rate vs. pH (Lead = 1)')
    ax[1].set_xlabel('pH')
    ax[1].set_ylabel('Infant Mortality Rate')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def part_b_iii_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]
    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph, "lead-pH"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    results = sm.OLS(y, x).fit()

    x_ph = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_ph]])
    results_ph = sm.OLS(y, x_ph).fit()

    f_test = results.compare_f_test(results_ph)
    print(results_ph.summary())
    print('F-statistic:', f_test[0])
    print('p-value:', f_test[1])


def part_b_iv_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]
    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph, "lead-pH"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    results = sm.OLS(y, x).fit()
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())
    p_values_pHxLead = results.pvalues["lead-pH"]
    print(f'p-value for pH X lead: {p_values_pHxLead:.4f}')


def part_b_v_solution():
    lead_mortality = LeadMortalityDataframe()
    avg_ph = lead_mortality.df[lead_mortality.column_name_ph].mean()
    std_ph = lead_mortality.df[lead_mortality.column_name_ph].std()
    ph_1_std_above = avg_ph + std_ph
    ph_1_std_below = avg_ph - std_ph
    print(f'Average pH: {avg_ph:.4f}')
    print(f'Standard Deviation of pH: {std_ph:.4f}')
    print(f'pH 1 Standard Deviation Above Average: {ph_1_std_above:.4f}')
    print(f'pH 1 Standard Deviation Below Average: {ph_1_std_below:.4f}')


def part_b_vi_solution():
    lead_mortality = LeadMortalityDataframe()
    std_infrate = lead_mortality.df[lead_mortality.column_name_infrate].std()

    print(f'Standard Deviation of Infant Mortality Rate: {std_infrate:.4f}')


def part_c_i_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]
    lead_mortality.df["lead-hardness"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_hardness]
    lead_mortality.df["ph-hardness"] = lead_mortality.df[lead_mortality.column_name_ph] * \
        lead_mortality.df[lead_mortality.column_name_hardness]

    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph,
                                           lead_mortality.column_name_hardness,
                                           "lead-pH",
                                           "lead-hardness",
                                           "ph-hardness"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())


def part_c_ii_solution():
    lead_mortality = LeadMortalityDataframe()
    lead_mortality.df["lead-pH"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_ph]
    lead_mortality.df["lead-mom_rate"] = lead_mortality.df[lead_mortality.column_name_lead] * \
        lead_mortality.df[lead_mortality.column_name_mom_rate]
    lead_mortality.df["ph-mom_rate"] = lead_mortality.df[lead_mortality.column_name_ph] * \
        lead_mortality.df[lead_mortality.column_name_mom_rate]

    x = sm.add_constant(lead_mortality.df[[lead_mortality.column_name_lead,
                                           lead_mortality.column_name_ph,
                                           lead_mortality.column_name_mom_rate,
                                           "lead-pH",
                                           "lead-mom_rate",
                                           "ph-mom_rate"]])
    y = lead_mortality.df[lead_mortality.column_name_infrate]
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())


if __name__ == '__main__':
    part_c_ii_solution()
