import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseRawDataPipeline(ABC):
    def __init__(self, input_data_path, output_data_path):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        ...

    def save(self, transformed_data: pd.DataFrame):
        transformed_data.to_pickle(self.output_data_path)

    def run(self):
        raw = self.load()
        transformed = self.transform(raw)
        self.save(transformed)


class SensorCyclePipeline(BaseRawDataPipeline):
    @staticmethod
    def add_exponential_weighted_features(df, sensor_cols, spans=[1000, 100000], funcs=['mean', 'std'], min_rows=100):
        result = []
    
        def ewm_with_min_rows(x, span, func_name):
            if func_name == 'mean':
                ewm_result = x.ewm(span=span, adjust=False).mean()
            elif func_name == 'std':
                ewm_result = x.ewm(span=span, adjust=False).std()
            else:
                return x * np.nan  # placeholder for unsupported func
            # mask early rows with insufficient count
            mask = x.expanding().count() < min_rows
            return ewm_result.where(~mask)
    
        for span in spans:
            for func in funcs:
                # Apply transform per column to preserve independence
                for col in sensor_cols:
                    col_feat = df.groupby("CycleID")[col].transform(
                        lambda x: ewm_with_min_rows(x, span=span, func_name=func)
                    )
                    col_feat.name = f"{col}_exp_weighted_{func}_span{span}"
                    result.append(col_feat)
    
        return pd.concat(result, axis=1)
        
    @staticmethod
    def add_z_scores(df, sensor_cols, spans=[10000]):
        z_scores = []
        for col in sensor_cols:
            for span in spans:
                mean_col = f"{col}_exp_weighted_mean_span{span}"
                std_col = f"{col}_exp_weighted_std_span{span}"
    
                if mean_col in df.columns and std_col in df.columns:
                    z = (df[col] - df[mean_col]) / df[std_col]
                    z.name = f"{col}_zscore_span{span}"
                    z_scores.append(z)
    
        if z_scores:
            return pd.concat(z_scores, axis=1)
        else:
            # Return empty DataFrame if no z-scores were created
            return pd.DataFrame(index=df.index)



    def load(self) -> pd.DataFrame:
        cols = ['timestamp'] + [f'sensor_{i:02d}' for i in range(52) if i != 15] + ['machine_status']
        return pd.read_csv(self.input_data_path, 
                           usecols=cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        #df['CycleID'] = (df['machine_status'] != df['machine_status'].shift(1)).cumsum() # first row is NaN
        df['CycleID'] = (df['machine_status'] != df['machine_status'].shift(1).fillna(df['machine_status'].iloc[0])).cumsum()

    
        # Rename sensor columns to sensorXX_value
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        rename_dict = {col: f"{col}_value" for col in sensor_cols}
        df = df.rename(columns=rename_dict)
    
        raw_sensor_cols = [f"{col}_value" for col in sensor_cols]
    
        # Decimal length extraction
        decimals = df[raw_sensor_cols].astype(str).apply(
            lambda x: x.str.split('.').str[1].str.len().astype(float)
        )
        decimals.columns = [f'{col}_decimals' for col in raw_sensor_cols]
    
        # Add exponential weighted features
        exp_weighted_feats = self.add_exponential_weighted_features(df, raw_sensor_cols, spans=[10000], funcs=['mean', 'std'])

        # Combine all features
        df = pd.concat([df, decimals, exp_weighted_feats], axis=1)

        z_scores =  self.add_z_scores(df, raw_sensor_cols, spans=[10000])
        
        df = pd.concat([df, z_scores], axis=1)
    
        return df
