"""
File: runner.py
------------------
Runner script to train the model. This is the script that calls the other modules.
Execute this one to execute the program! 
"""


import configs
import maml_trainer
import argparse
import warnings
import pdb 


def main(args):
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config
    cfg = getattr(configs, config_class)
    exp = cfg.experiment(
        trainloader = cfg.trainloader,
        # meta_optimizer = cfg.meta_optimizer,
        model = cfg.model,
        criterion = cfg.criterion,
        num_inner_steps = cfg.num_inner_steps,
        inner_lr = cfg.inner_lr,
        logger = cfg.logger
    )
    exp.train(num_epochs=15)



if __name__ == '__main__':
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    args = parser.parse_args()
    main(args)