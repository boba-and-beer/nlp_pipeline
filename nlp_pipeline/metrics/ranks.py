"""
Ranking Metrics. 
Current Implementations: 
- SpearmanRho
"""
from fastai.text import *
from fastai.callbacks.tracker import *

class SpearmanRho(Callback):
    """
    FastAI callback
    Example: 
    learner = Learner(databunch, callback_fns=)
    """

    def on_epoch_begin(self, **kwargs):
        # Probably do not need the below.
        self.spearman_rho = 0
        self.spearman_scores = []
        # Calculate the output scores
        self.targets = None
        self.outputs = None

    def on_batch_end(self, last_output, last_target, **kwargs):
        if isinstance(last_output, tuple):
            last_output = last_output[0]
        if self.targets is None:
            self.targets = last_target
        else:
            self.targets = torch.cat((self.targets, last_target))
        if self.outputs is None:
            self.outputs = last_output
        else:
            self.outputs = torch.cat((self.outputs, last_output))

    def on_epoch_end(self, last_output, last_target, last_metrics, **kwargs):
        if isinstance(last_output, tuple):
            last_output = last_output[0]
        for i in range(self.outputs.shape[1]):
            # Calculate spearman score across the differerent metrics
            spearman_score = spearmanr(
                self.outputs.cpu().permute(1, 0)[i], self.targets.cpu().permute(1, 0)[i]
            )
            self.spearman_scores.append(spearman_score[0])
        self.spearman_rho = np.nanmean(self.spearman_scores)
        nan_count = sum([np.isnan(x) for x in self.spearman_scores]) / len(
            self.spearman_scores
        )
        if nan_count > 0:
            print("The number of nan in the spearman scores is: " + str(nan_count))
        return add_metrics(last_metrics, self.spearman_rho)
