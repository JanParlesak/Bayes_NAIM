
from experiment import * 


config = {
    "mode": 'classification',
    "dropout_rate": tune.choice([2, 4, 8, 16]),
    "feature_dropout_rate": tune.choice([2, 4, 8, 16]),
    "batch_size": tune.choice([2, 4, 8, 16]),
    "prior_scale" : 0.1,
    "learning_rate": tune.choice([2, 4, 8, 16]),
    "n_epochs" : 1000,
    "n_samples": 1,
    "n_post_samples": 10,
    "kl_weight" : 0.01,
    "num_trials": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main(config, gpus_per_trial=1):
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_bnaim),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="AUC_PR",
            mode="max",
            scheduler=scheduler,
            num_samples=config["num_trials"],
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("AUC_PR", "max")

    print(f"Best trial config: {best_result.config}")
    print(f"Best AUC_PR: {best_result.metrics['AUC_PR']}, Best AUC: {best_result.metrics['AUC']}, Best Accuracy: {best_result.metrics['Balanced Accuracy']}, Best Recall: {best_result.metrics['val_recall']}, Best Precision: {best_result.metrics['val_precision']}")

    get_test_predictions(best_result)

    

main(config, gpus_per_trial=1 if torch.cuda.is_available() else 0)

