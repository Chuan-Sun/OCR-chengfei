train_opt = {
    'data_dir': '../training_data',
    'input_size': [1, 32, 160],
    'label_dic': {},
    'label_dim': -1,
    'max_label_length': 25,
    'model': '',  # 'checkpoints/model_epoch_10.pth',
    'n_epochs': 90,
    'batch_size': 128,
    'lr': 1e-3
}