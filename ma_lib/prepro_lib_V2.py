import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer

class DataFrameAnalyzer:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("L'objet doit être un DataFrame pandas.")
        self.df = df

    def split_df(self, keep='num'):
        categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if keep == 'num':
            if not categorical_cols:
                return self.df
            else:
                return self.df[numerical_cols]
        elif keep == 'obj':
            if not numerical_cols:
                return self.df
            else:
                return self.df[categorical_cols]
        elif keep == 'all':
            return self.df[numerical_cols], self.df[categorical_cols]
        else:
            raise ValueError("Argument keep incorect ! valueur possible (num, obj, all)")

    def where_are_nan(self, num = 0):
        dico_nan = {}
        for col in self.df.columns:
            a = self.df[col].isna().sum()/len(self.df)
            if a > num:
                dico_nan[col] = a
        return dico_nan

    def col_corr(self, y_col):
        
        if y_col is None:
            raise ValueError("Argument y_col manquant")

        num_df = self.split_df(keep = 'num')
        score_dico = {}

        for col in num_df.columns:
            _, correlation_score = self.test_colinearity(col, y_col)
            score_dico[col] = correlation_score
        
        corr_df = pd.DataFrame(list(score_dico.items()), columns=['column', 'correlation'])
        corr_df.rename(columns={'correlation': f'corr_with_{y_col}'}, inplace=True)

        corr_df['abs_correlation'] = corr_df[f'corr_with_{y_col}'].abs()
        corr_df.sort_values('abs_correlation', ascending=False, inplace=True)
        corr_df.drop(columns=['abs_correlation'], inplace=True)

        return corr_df


    def test_colinearity(self, col1, col2):

        if self.df[col1].isnull().any():
            imputer = KNNImputer()
            self.df[col1] = pd.Series(imputer.fit_transform(self.df[[col1]]).ravel())
            print(f'colonne: {col1} a été imputée car elle contenait des NaN')
        if self.df[col2].isnull().any():
            imputer = KNNImputer()
            self.df[col2] = pd.Series(imputer.fit_transform(self.df[[col2]]).ravel())
            print(f'colonne: {col2} a été imputée car elle contenait des NaN')

        methods = ['pearson', 'spearman', 'kendall']
        best_score = 0
        best_method = ''

        for method in methods:
            if method == 'pearson':
                corr, _ = stats.pearsonr(self.df[col1], self.df[col2])
            elif method == 'spearman':
                corr, _ = stats.spearmanr(self.df[col1], self.df[col2])
            elif method == 'kendall':
                corr, _ = stats.kendalltau(self.df[col1], self.df[col2])
            
            if abs(corr) > abs(best_score):
                best_score = corr
                best_method = method

        return best_method, best_score


    def drop_collinear_columns(self, y_col, threshold=0.7):
        if y_col is None:
            raise ValueError("Argument y_col manquant")

        num_df = self.split_df(keep='num')
        corr_df = self.col_corr(y_col)

        dropped_cols_colinear = []

        for col1 in num_df.columns:
            for col2 in num_df.columns:
                if col1 != col2 and col1 != y_col and col2 != y_col:
                    _, score = self.test_colinearity(col1, col2)
                    if abs(score) > threshold:
                        if corr_df.loc[corr_df['column'] == col1, f'corr_with_{y_col}'].values[0] > corr_df.loc[corr_df['column'] == col2, f'corr_with_{y_col}'].values[0]:
                            if col2 not in dropped_cols_colinear:
                                num_df.drop(columns=[col2], inplace=True)
                                dropped_cols_colinear.append(col2)
                        else:
                            if col1 not in dropped_cols_colinear:
                                num_df.drop(columns=[col1], inplace=True)
                                dropped_cols_colinear.append(col1)

        print("Colonnes supprimées car elles sont trop colinéaires : ", dropped_cols_colinear)

        return num_df