import os
import re
from openai import OpenAI
from pathlib import Path
from src.file_handler import read_file

class MessageClassifier:
    """A classifier that uses a local LLM to categorize messages."""

    def __init__(self, base_url: str, api_key: str = "lm-studio"):
        """
        Initializes the classifier and connects to the LLM server.

        Args:
            base_url (str): The base URL of the LM Studio server.
            api_key (str): The API key for the server (default is 'lm-studio').
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)



    def classify_message(self, messages: list[dict]) -> list[str]:
        """
        Sends a list of messages to the LLM and returns the parsed classification.

        Args:
            messages (list[dict]): A list of message dictionaries to send to the API.

        Returns:
            list[str]: The predicted categories or an error message.
        """
        try:
            completion = self.client.chat.completions.create(
                model="local-model",  # This value is ignored by LM Studio
                messages=messages,
                temperature=0.1,
            )
            response_text = completion.choices[0].message.content.strip()

            # Post-process the response to remove the model's thought process
            if '</think>' in response_text:
                # Take the part after the last thought block
                clean_response = response_text.split('</think>')[-1]
            else:
                clean_response = response_text

            # Split the cleaned string by comma and strip whitespace from each item
            return [category.strip() for category in clean_response.strip().split(',')]
        except Exception as e:
            print(f"An error occurred while communicating with the LLM: {e}")
            return ["Error: Classification failed"]
