import pandas as pd
import numpy as np
from constants import CODE_TO_NAME, ZERO_VAR_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN
from config.config import RAW_DATA_DIR

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.set_option('future.no_silent_downcasting', True)




class DataPreprocessor:
    def __init__(self):
        self.zero_var_columns = None

    def preprocess_pipeline(self, data, mode='eval'):
        data = data.copy()
        self.validate_required_columns(data)
        data = self.delete_redundant_columns(data)
        data = self.add_deltas_abs(data)
        data = self.add_deltas_rel(data)
        data = self.add_financial_ratios(data)
        data = self.cap_outliers(data)
        if mode=='eval':
            if self.zero_var_columns is None:
                raise ValueError('You must train model with DataPreprocessor at first')
            data, zero_var_columns = self.delete_zero_var_columns(data,  mode=mode,zero_var_columns=self.zero_var_columns)
        elif mode=='train':
            data, zero_var_columns = self.delete_zero_var_columns(data, mode='train')
            self.zero_var_columns = zero_var_columns

        return data

    def validate_required_columns(self, data):
        missing = set(REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def delete_redundant_columns(self, data):
        return data[REQUIRED_COLUMNS]


    def add_deltas_abs(self, data):
        for code in CODE_TO_NAME:
            if f'P{code}_B' in data.columns and f'P{code}_E' in data.columns:
                data[f'P{code}_delta_abs'] = data[f'P{code}_E'] - data[f'P{code}_B']
        return data

    def add_deltas_rel(self, data):
        for code in CODE_TO_NAME:
            if f'P{code}_B' in data.columns and f'P{code}_E' in data.columns:
                data[f'P{code}_delta_rel'] = np.where(data[f'P{code}_B']!=0,
                                                         (data[f'P{code}_E'] - data[f'P{code}_B']) / data[f'P{code}_B'],
                                                         0)
        return data

    def add_financial_ratios(self, data):


        # 1. Equity to Assets Ratio
        data['equity_to_assets_ratio'] = np.where(
            data['P1300_E'].abs() > 0,
            data['P1300_E'] / data['P1700_E'].replace(0, np.nan),
            np.nan
        )

        # 2. Debt to Assets Ratio
        data['debt_to_assets_ratio'] = np.where(
            data['P1700_E'].abs() > 0,
            (data['P1400_E'] + data['P1500_E']) / data['P1700_E'],
            np.nan
        )

        # 3. Current Ratio
        data['current_ratio'] = np.where(
            data['P1500_E'].abs() > 0,
            data['P1200_E'] / data['P1500_E'],
            np.nan
        )

        # 4. Quick Ratio
        data['quick_ratio'] = np.where(
            data['P1500_E'].abs() > 0,
            (data['P1250_E'] + data['P1230_E']) / data['P1500_E'],
            np.nan
        )

        # 5. Cash Ratio
        data['cash_ratio'] = np.where(
            data['P1500_E'].abs() > 0,
            data['P1250_E'] / data['P1500_E'],
            np.nan
        )

        # 6. ROA
        avg_assets = (data['P1600_B'] + data['P1600_E']) / 2
        data['roa'] = np.where(
            avg_assets.abs() > 0,
            data['P2400_E'] / avg_assets,
            np.nan
        )

        # 7. ROS
        data['ros'] = np.where(
            data['P2110_E'].abs() > 0,
            data['P2400_E'] / data['P2110_E'],
            np.nan
        )

        # 8. ROE
        avg_equity = (data['P1300_B'] + data['P1300_E']) / 2
        data['roe'] = np.where(
            avg_equity.abs() > 0,
            data['P2400_E'] / avg_equity,
            np.nan
        )

        # 9. Asset Turnover
        data['asset_turnover'] = np.where(
            avg_assets.abs() > 0,
            data['P2110_E'] / avg_assets,
            np.nan
        )

        # 10. Receivables Turnover
        avg_receivables = (data['P1230_B'] + data['P1230_E']) / 2
        data['receivables_turnover'] = np.where(
            avg_receivables.abs() > 0,
            data['P2110_E'] / avg_receivables,
            np.nan
        )

        return data


    def get_acceptable_range(self, values, threshold=0.01):

        min_value = values.quantile(threshold)
        max_value = values.quantile(1 - threshold)
        if np.isnan(min_value) or np.isnan(max_value):
            return values.min(), values.max()
        return min_value, max_value

    def cap(self, x, min_value, max_value):
        if x < min_value:
            return min_value
        elif x > max_value:
            return max_value
        return x

    def cap_outliers(self, data, target='BANKR'):
        for col in data.columns:
            if col != target:
                min_value, max_value = self.get_acceptable_range(data[col], threshold=0.01)
                data[col] = data[col].clip(lower=min_value, upper=max_value)
        return data

    def delete_zero_var_columns(self, data, mode='eval', zero_var_columns=None):
        if mode=='eval':
            return data.drop(zero_var_columns, axis=1), zero_var_columns
        elif mode=='train':
            columns_to_delete = []
            for col in data.columns:
                if (col != TARGET_COLUMN) and (data[col].std() == 0):
                    columns_to_delete.append(col)
            if len(columns_to_delete) == 0:
                return data, columns_to_delete
            else:
                return data.drop(columns_to_delete, axis=1), columns_to_delete
        return None


if __name__ == "__main__":
    data = pd.read_csv(RAW_DATA_DIR + '/test.csv')
    data = DataPreprocessor().preprocess_pipeline(data, zero_var_columns=ZERO_VAR_COLUMNS)
    print(data.head())
