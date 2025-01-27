from flask import Flask, request, jsonify
import requests
import os
import rag
import threading
import json
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path='secrets.env')

# Initialize Flask app
app = Flask(__name__)

# Slack configuration (replace with your actual token and signing secret)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

processed_message_ids = set()

logging.basicConfig(
    filename="slack_events.log",  # Save logs to a file
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handles Slack events like incoming messages."""
    data = request.json
    
    logging.info("Incoming Slack Event: %s", json.dumps(data, indent=2))

    # Slack URL verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
#____________________________________________________
    if "event" in data:
        event = data["event"]
        event_type = event.get("type")
        user = event.get("user")
        text = event.get("text")
        channel = event.get("channel")
        ts = event.get("ts")

        # Log specific event details
        logging.info("Event Type: %s", event_type)
        logging.info("User: %s", user)
        logging.info("Text: %s", text)
        logging.info("Channel: %s", channel)
        logging.info("Timestamp: %s", ts)
#________________________________________________________

    # Process message events
    if "event" in data:
        event = data["event"]
        if event.get("type") == "message" and not event.get("subtype"):

            message_id = event.get("client_msg_id")

            # Skip if message has already been processed
            if message_id in processed_message_ids or event.get("bot_id") is not None:
                return jsonify({"status": "duplicate_message_skipped"})

            # Add message ID to the processed set
            processed_message_ids.add(message_id)

            # Start background thread to process the message
            thread = threading.Thread(target=process_message_async, args=(event,))
            thread.start()

        # Acknowledge Slack immediately
        return jsonify({"status": "ok"})

    return jsonify({"status": "ok"})

def process_message_async(event):
    """Processes the Slack message asynchronously."""
    user_message = event.get("text")
    channel_id = event.get("channel")
    thread_ts = event.get("ts")

    rag_response = rag.init(user_message)

    # Send the generated response back to Slack
    send_message_to_slack(channel_id, rag_response, thread_ts)

def send_message_to_slack(channel_id, message, thread_ts):
    """Sends a message to Slack."""
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    payload = {
        "channel": channel_id,
        "text": message,
    }
    #Reply in thread 
    if thread_ts:
        payload["thread_ts"] = thread_ts
    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(port=2323)
