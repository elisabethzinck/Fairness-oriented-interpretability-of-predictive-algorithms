# Script is currently only holding code moved from cheXpert_neural_network
"""
output_path = 'data/CheXpert/predictions/cheXpert_nn_pred.csv'

cols_to_keep = ["patient_id", "y"] 
output_data = (dm.test_data.dataset_df[cols_to_keep]
    .assign(
        nn_prob = np.nan,
        nn_pred = np.nan
    ))



print('--- Testing and making predictions using best model ---')
    pl_model.model.eval()
    batch_start_idx = 0
    for batch in dm.test_dataloader():
        print(f"shape:{batch[0].shape}")
        nn_prob = (torch.sigmoid(pl_model.model
            .forward(batch[0]))
            .detach().numpy().squeeze())
        batch_end_idx = batch_start_idx + batch[0].shape[0]
        output_data.nn_prob.iloc[batch_start_idx:batch_end_idx] = nn_prob
        batch_start_idx = batch_end_idx

    output_data = output_data.assign(nn_pred = lambda x: x.nn_prob >= 0.5)
        
    acc = accuracy_score(output_data.nn_pred, output_data['y'])
    print(f'Final accuracy score: {acc}')
"""