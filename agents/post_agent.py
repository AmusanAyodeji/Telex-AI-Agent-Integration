from typing import List, Optional
from uuid import uuid4
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

from models.a2a import (
    A2AMessage, TaskResult, TaskStatus, Artifact,
    MessagePart, MessageConfiguration
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,                     # Log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()             # Also show them in the console
    ]
)

history: List[A2AMessage] = []

def generate_linkedin_post(API_KEY, brief: str) -> str:
  client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
  )
  completion = client.chat.completions.create(
    extra_body={},
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"""You are a linkedin post generator. When a user interacts with you be able to identify if the user wants you to turn their message into a linkedin post or if they are just trying to make a conversation and respond accordingly.

            Example:

            User: Hello
            Agent: Hello! How can I assist you today?
            User: Can you help me create a linkedin post?
            Agent: Sure, Just provide an explanation of your task and I'll do it right away

            In the case above, it was just a conversation, so respond reasonably. It is not required you use the exact wording but the point is to respond reasonably when the user doesn't provide an actual brief for a linkedin post. the User and Agent part of the exmaple is just to show the person typing and its not to be included in the actual response. 

            If a user will provides you with a brief of a task they've completed and all the steps they took and other details about the project. Your task is to generate a linkedin post that is engaging and professional. The post should be between 100-150 words. Use appropriate hashtags and a call to action at the end of the post. Here is the brief: {brief}"""
          }
        ]
      }
    ]
  )

  return completion.choices[0].message.content


class PostAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    
    async def process_messages(
        self,
        messages: List[A2AMessage],
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        config: Optional[MessageConfiguration] = None
    ) -> TaskResult:


        global history

        logging.info(f"Processing message: {messages}")

        context_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())

        user_message = messages[-1] if messages else None
        if not user_message:
            raise ValueError("No message provided")
        
        message = user_message.parts[0].text if user_message.parts else ""
        
        try:
          linkedin_post = generate_linkedin_post(self.api_key, message)
        except Exception as e:
          linkedin_post = f"Error generating post: {str(e)}"

        response_message = A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=linkedin_post)],
            taskId=task_id
        )

        # Build history
        history += messages + [response_message]

        result = TaskResult(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(
                state="completed",
                message=response_message
            ),
            history=history
        )

        logging.info(f"sending message: {messages}")

        return result