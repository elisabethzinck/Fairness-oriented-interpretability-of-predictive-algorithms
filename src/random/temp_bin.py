    # def l1_get_data(self, w_fp = None):
    #     """Get data used for first layer of evaluation
        
    #     Args:
    #         w_fp (float): False positive error weight
    #     """
    #     if w_fp is None:
    #         w_fp = self.w_fp
    #     df = (pd.DataFrame(self.cm)
    #         .T.reset_index()
    #         .rename(columns = {'index': 'group'})
    #         .assign(
    #             n = lambda x: x.TP + x.FN + x.FP + x.TN,
    #             percent_positive = lambda x: (x.TP + x.FN)/x.n*100,
    #             WMR = lambda x: (w_fp*x.FP + (1-w_fp)*x.FN)/(2*x.n)))
    #     WMR_min = min(df.WMR)
    #     df['unfair'] = (df.WMR-WMR_min)/abs(WMR_min)*100

    #     # Make table pretty
    #     cols_to_keep = ['group', 'unfair', 'WMR', 'n', 'percent_positive']
    #     digits = {'unfair': 1, 'WMR': 3, 'percent_positive': 1}
    #     df = (df[cols_to_keep]
    #         .round(digits))

    #     return df