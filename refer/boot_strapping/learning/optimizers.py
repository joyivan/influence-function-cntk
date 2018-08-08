import os
import time
import numpy as np
from abc import abstractmethod
import cntk as C
from learning.utils import plot_learning_curve
from torch.utils.data import DataLoader


class Optimizer(object):
    """Base class for gradient-based optimization algorithms."""

    def __init__(self, model, train_set, evaluator, val_set=None, fold_idx=0, trainval_ratio=1.0, **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet, the model to be learned.
        :param train_set: DataSet, training set to be used. (which should be lazy)
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - batch_size: int, batch size for each iteration.
            - num_epochs: int, total number of epochs for training.
            - init_learning_rate: float, initial learning rate.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        self.fold_idx = fold_idx
        self.trainval_ratio = trainval_ratio
        
        # Spec hyperparameters
        self.num_workers = kwargs.pop('num_workers', 4)

        # Training hyperparameters
        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)
        self._reset()

        self.learner = self._learner()
        self.trainer = C.Trainer(self.model.logits, self.model.loss,
                                 [self.learner])

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1
        self.num_bad_epochs = 0    # number of bad epochs, where the model is updated without improvement.
        self.best_score = self.evaluator.worst_score    # initialize best score with the worst one
        self.curr_learning_rate = self.init_learning_rate    # current learning rate

    @abstractmethod
    def _learner(self, **kwargs):
        """
        cntk.learners.Learner for a gradient update.
        This should be implemented, and should not be called manually.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate (if needed) on every epoch, by its own schedule.
        This should be implemented, and should not be called manually.
        """
        pass

    def train(self, model_name='model1', save_dir='/Data/checkpts', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param save_dir: str, the directory to save the learned weights of the model.
        :param details: bool, whether to return detailed results.
        :param verbose: bool, whether to print details during training.
        :param kwargs: dict, extra arguments containing training hyperparameters.
        :return train_results: dict, containing detailed results of training.
        """
        train_results = dict()    # dictionary to contain training(, evaluation) results and details
        dataloader = DataLoader(self.train_set, self.batch_size, shuffle=True,\
                num_workers=self.num_workers)
        #train_size = self.train_set.num_examples
        #num_steps_per_epoch = train_size // self.batch_size
        #num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(self.num_epochs))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # Start training loop
        for epoch in range(self.num_epochs):
            for X, y_true in dataloader:
                X = X.numpy(); y_true = y_true.numpy()
                y_pred = self.model.pred.eval({self.model.X: X})
                self.trainer.train_minibatch({self.model.X: X, self.model.y: y_true})
                step_loss =self.trainer.previous_minibatch_loss_average
                step_losses.append(step_loss)

            # why outer loop?
            step_score = self.evaluator.score(y_true, y_pred)
            step_scores.append(step_score)
            #print('train{}'.format(np.mean(y_pred==np.ones_like(y_pred))))

            # If validation set is initially given, use it for evaluation
            if self.val_set is not None:
                # Evaluate model with the validation set
                temp_scores = []
                val_dataloader = DataLoader(self.val_set, self.batch_size, shuffle=False,\
                        num_workers=self.num_workers)
                for X, y in val_dataloader:
                    X = X.numpy(); y = y.numpy()
                    y_pred = self.model.pred.eval({self.model.X: X})
                    temp_score = self.evaluator.score(y, y_pred)
                    temp_scores.append(temp_score)
                eval_score = np.mean(temp_scores)
                eval_scores.append(np.mean(eval_score))

                if verbose:
                    # Print intermediate results
                    print('[epoch {}]\tloss: {:.6f} |Train {}: {:.6f} |Eval {}: {:.6f} |lr: {:.6f}'\
                          .format(self.curr_epoch, step_loss,
                                  self.evaluator.name, step_score,
                                  self.evaluator.name, eval_score, self.curr_learning_rate))
                    #print('eval{}'.format(np.mean(y_pred==np.ones_like(y_pred))))
                    # Plot intermediate results
                    plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                        ylabel=self.evaluator.name, mode=self.evaluator.mode,
                                        img_dir=save_dir)
                curr_score = eval_score

            # else, just use results from current minibatch for evaluation
            else:
                if verbose:
                    # Print intermediate results
                    print('[epoch {}]\tloss: {} |Train {}: {:.6f} |lr: {:.6f}'\
                          .format(self.curr_epoch, step_loss,
                                  self.evaluator.name, step_score, self.curr_learning_rate))
                    # Plot intermediate results
                    plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                        ylabel=self.evaluator.name, mode=self.evaluator.mode,
                                        img_dir=save_dir)
                curr_score = step_score

            # Keep track of the current best model,
            # by comparing current score and the best score
            if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                self.best_score = curr_score
                self.num_bad_epochs = 0
                # Save current weights <- add fold and trainval info
                self.model.logits.save(os.path.join(save_dir,'noisy',\
                        'model_fold_{}_trainval_ratio_{}.dnn'.format(self.fold_idx, self.trainval_ratio)))
            else:
                self.num_bad_epochs += 1

            self._update_learning_rate(**kwargs)
            self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} {}: {}'.format('evaluation' if eval else 'training',
                                          self.evaluator.name, self.best_score))
        print('Done.')


        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results


class MomentumOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _learner(self, **kwargs):
        """
        cntk.learners.Learner for a gradient update.
        :param kwargs: dict, extra arguments for optimizer.
            - momentum: float, the momentum coefficient.
            - weight_decay: float, L2 weight decay regularization coefficient.
        :return cntk.learners.Learner.
        """
        momentum = kwargs.pop('momentum', 0.9)
        weight_decay = kwargs.pop('weight_decay', 0.0000)

        update_vars = self.model.logits.parameters
        lr_schedule = C.learning_parameter_schedule(self.curr_learning_rate)
        mm_schedule = C.momentum_schedule(momentum)
        return C.momentum_sgd(update_vars, lr_schedule, mm_schedule,
                              l2_regularization_weight=weight_decay)

    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate, when evaluation score plateaus.
        :param kwargs: dict, extra arguments for learning rate scheduling.
            - learning_rate_patience: int, number of epochs with no improvement
                                      after which learning rate will be reduced.
            - learning_rate_decay: float, factor by which the learning rate will be updated.
            - eps: float, if the difference between new and old learning rate is smaller than eps,
                   the update is ignored.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # Decay learning rate only when the difference is higher than epsilon.
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
                # Update learner's learning rate
                new_lr_schedule = C.learning_parameter_schedule(self.curr_learning_rate)
                self.learner.reset_learning_rate(new_lr_schedule)
                print('New learning rate: {:.6f}'.format(self.learner.learning_rate()))
            self.num_bad_epochs = 0
