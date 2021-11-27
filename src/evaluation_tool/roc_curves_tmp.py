# To do: Documentation

    def calculate_roc_curve(classifier, sens_grps):
        roc_list = []
        for grp in sens_grps:
            data_grp = classifier[classifier.a == grp]
            fpr, tpr, thresholds = roc_curve(
                y_true = data_grp.y, 
                y_score = data_grp.r)
            roc_grp = (
                pd.DataFrame({
                    'fpr': fpr, 
                    'tpr': tpr, 
                    'threshold': thresholds,
                    'sens_grp': grp}))
            roc_list.append(roc_grp)
        roc = pd.concat(roc_list).reset_index(drop = True)  
        return roc  

    n_boot = 100
    n = fair_anym.classifier.shape[0]
    roc_list = []
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        indices = rng.randint(0, n, n)
        boot = fair_anym.classifier.iloc[indices]
        roc_boot = calculate_roc_curve(boot, fair_anym.sens_grps).assign(i = i)
        roc_list.append(roc_boot)

    roc_final = pd.concat(roc_list).reset_index(drop = True)
    #%%
    tmp = roc_final.query("sens_grp == 'A'")
    sns.lineplot(
                x = 'fpr', y = 'tpr', hue = 'i',
                data = tmp,
                estimator = None, alpha = 0.8,
                zorder = 1, legend = False)