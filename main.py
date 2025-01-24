from flask import Flask, request, jsonify
import requests
import os
import rag
from dotenv import load_dotenv

load_dotenv(dotenv_path='secrets.env')

# Initialize Flask app
app = Flask(__name__)

# Slack configuration (replace with your actual token and signing secret)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handles Slack events like incoming messages."""
    data = request.json
    
    # Slack URL verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Process message events
    if "event" in data and data["event"]["type"] == "message":
        user_message = data["event"]["text"]  # Extract user message
        channel_id = data["event"]["channel"]  # Get Slack channel ID

        # Main RAG Call 
        if user_message.strip().startswith("/askBot"):
            query = user_message.strip().replace("/askBot", "", 1).strip()  # Remove '/askBot' prefix
            if query:
                rag_response = rag.init(user_message)

        # Send response back to Slack
        send_message_to_slack(channel_id, rag_response)

    return jsonify({"status": "ok"})


def send_message_to_slack(channel_id, message):
    """Sends a message to Slack."""
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    payload = {
        "channel": channel_id,
        "text": message,
    }
    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(port=2323)
