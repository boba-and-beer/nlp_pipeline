"""
Not really sure where these plots should be going so storing them in utils for now.
"""


def plot_losses_and_lr(
    self,
    skip_start: int = 10,
    skip_end: int = 0,
    show_moms=True,
    return_fig: bool = None,
) -> Optional[plt.Figure]:
    """
    Add this to a method to the learner in order to plot the most recent loses and lr. 
    This needs to be added manually after eaach record call.
    """
    if show_moms:
        fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    else:
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    losses = self._split_list(self.losses, skip_start, skip_end)
    iterations = self._split_list(range_of(self.losses), skip_start, skip_end)
    ax[0].plot(iterations, losses, label="Train")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Batches processed")
    "Plot learning rate, `show_moms` to include momentum."
    lrs = self._split_list(self.lrs, skip_start, skip_end)
    iterations = self._split_list(range_of(self.lrs), skip_start, skip_end)
    if show_moms:
        moms = self._split_list(self.moms, skip_start, skip_end)
        # fig, axs = plt.subplots(1,2, figsize=(12,4))
        ax[1].plot(iterations, lrs)
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("Learning Rate")
        ax[2].plot(iterations, moms)
        ax[2].set_xlabel("Iterations")
        ax[2].set_ylabel("Momentum")
    else:
        # fig, ax = plt.subplots()
        ax[1].plot(iterations, lrs)
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("Learning Rate")
    if ifnone(return_fig, defaults.return_fig):
        return fig
