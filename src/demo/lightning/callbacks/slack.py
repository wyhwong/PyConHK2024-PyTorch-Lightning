from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

import demo.logger
from demo.slack.client import SlackClient


local_logger = demo.logger.get_logger(__name__)


class MonitorTrainingOnSlack(Callback):
    """Monitor training on slack

    This callback will do the following:
    1. Log precision of model on validation set to Slack every n epochs.
    2. Stop training if requested (received text messsage "stop" from Slack).
    """

    def __init__(
        self,
        slack_client: SlackClient,
        log_every_n_epoch: int = 1,
    ) -> None:
        """Initialize MonitorTrainingOnSlack

        Args:
            slack_client (SlackClient): Slack client object.
            log_every_n_epoch (int, optional): Log the precision to Slack every n epochs.
                Defaults to 1.
        """

        self._log_every_n_epoch = log_every_n_epoch
        self._slack_client = slack_client
        self._slack_client.start_socket_mode()

    def _should_skip_check(self, trainer) -> bool:
        """Check if the sanity check should be skipped"""

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends"""

        if self._should_skip_check(trainer):
            return

        # Log the validation loss to slack
        best_precision = pl_module.best_precision
        current_epoch = pl_module.current_epoch

        if current_epoch % self._log_every_n_epoch == 0:
            self._slack_client.post_message(f"Epoch {current_epoch} - Best Precision: {best_precision}")

        # Stop training if requested.
        if self._slack_client.stop_training:
            local_logger.warning("Stop training requested. Training will stop in the end of current epoch.")
            trainer.should_stop = True

            self._slack_client.post_message(f"Training stopped at epoch {current_epoch} due to stop training request.")
            self._slack_client.terminate()
            local_logger.info("Slack client terminated due to stop training request.")
