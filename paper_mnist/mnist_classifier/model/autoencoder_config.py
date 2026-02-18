rep = 1

conf = dict(
    classifier = dict(
        lr = 5e-4,
        bs = 32,
        num_workers = 0,
        hidden_layers = [500, 400, 300],
        drop_out_p = 0.5,
        save_path = f'./paper_mnist/mnist_classifier/model/do_classifier/rep_{rep}',
        patience = 10,
        use_batch_norm = True,
        no_act_1st_layer = False,
        logger = dict(
            # name = 'tb_logs',
            version = f'bs_128_lr_5em4_3_do_5_bn_T_{rep}'
        ),
        trainer = dict(
            devices = 0,
            max_epochs = 200,
            accelerator = 'cpu',
            strategy = 'ddp', #'ddp_find_unused_parameters_true',
            refresh_rate = 50
        ),
        start_with_init_weights = True,
        save_initial_weights = False,
        save_initial_weights_path = f'./paper_mnist/mnist_classifier/model/replicates/clf_weights_hs_300_bn_T_mlp_{rep}.pth'
    ),
    autoencoder = dict(
        lr = 2e-3,
        bs = 32,
        num_workers = 0,
        encoder_layers = [500, 400, 300],
        decoder_layers = [400, 500],
        drop_out_p = .0,
        save_path = f'./paper_mnist/mnist_classifier/model/base_classifier/rep_{rep}',
        patience = 10,
        use_batch_norm = True,
        logger = dict(
            # name = 'tb_logs',
            version = f'ae_bs_128_lr_2em3_hs_300_{rep}'
        ),
        trainer = dict(
            devices = 0,
            max_epochs = 200,
            accelerator = 'cpu',
            strategy = 'ddp',
            refresh_rate = 50
        ),
        load_classifier_path = f'./paper_mnist/mnist_classifier/model/base_classifier/rep_{rep}/mnist_clf_epoch=068-val_loss=0.0006.ckpt'
    )
)