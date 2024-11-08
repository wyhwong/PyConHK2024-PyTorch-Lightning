import os
from threading import Thread
from typing import Any, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import WebClient

import demo.env
import demo.logger


local_logger = demo.logger.get_logger(__name__)


class SlackClient:
    """Slack client"""

    def __init__(
        self,
        slack_app_token: str = demo.env.SLACK_APP_TOKEN,
        slack_bot_token: str = demo.env.SLACK_BOT_TOKEN,
    ):
        """Constructor"""

        self._app_token = slack_app_token
        self._bot_token = slack_bot_token

        self._web_client = WebClient(token=self._bot_token)
        self._subscriber_thread: Optional[Thread] = None

        # For training
        self.stop_training = False

    def start_socket_mode(self):

        app = App(token=self._bot_token)

        @app.event("message")
        def on_message(body):
            self.handle_message(body["event"])

        self._socket_mode_handler = SocketModeHandler(app, self._app_token)
        self._subscriber_thread = Thread(target=self._socket_mode_handler.start, daemon=True)
        self._subscriber_thread.start()

    def terminate(self):
        """Terminate the client"""

        if self._subscriber_thread:
            self._socket_mode_handler.close()
            self._subscriber_thread.join()

    def handle_message(self, event: dict[str, Any]):
        """Handle message"""

        text, user, channel = event["text"], event["user"], event["channel"]
        local_logger.info("Message received: %s from %s in %s", text, user, channel)

        if text == "stop":
            self.stop_training = True
            local_logger.warning("Set stop_training to True. Training will stop in the end of current epoch.")

    def post_message(
        self,
        text: str,
        channel_id: str = demo.env.SLACK_DEFAULT_CHANNEL_ID,
    ):
        """Post message"""

        local_logger.info("Sending message to Slack: %s", text)

        response = self._web_client.chat_postMessage(channel=channel_id, text=text)
        is_sent = response["ok"]

        local_logger.info("Message sent: %s", is_sent)

    def post_image(
        self,
        filepath: str,
        channel_id: str = demo.env.SLACK_DEFAULT_CHANNEL_ID,
    ):
        """Post image"""

        local_logger.info("Sending file to Slack: %s", filepath)

        title = os.path.basename(filepath).split(".")[0]
        response = self._web_client.files_upload_v2(title=title, file=filepath, channel=channel_id)
        is_sent = response["ok"]

        local_logger.info("File sent: %s", is_sent)
