import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Choice, Message, Request, Role, Stage
from pydantic import StrictStr

_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        """Ensure UMS conversation context, call UMS agent, and return output."""

        conversation_id = self.__get_ums_conversation_id(request)
        if not conversation_id:
            conversation_id = await self.__create_ums_conversation()
            choice.set_state({_UMS_CONVERSATION_ID: conversation_id})

        last_message = request.messages[-1]
        user_message = last_message.content or ""
        if additional_instructions:
            user_message = (
                f"{user_message}\n\n"
                f"Additional instructions:\n{additional_instructions}"
            )

        content = await self.__call_ums_agent(
            conversation_id=conversation_id,
            user_message=user_message,
            stage=stage,
        )

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""

        for message in request.messages:
            custom_content = message.custom_content
            if not custom_content or not custom_content.state:
                continue
            if (
                isinstance(custom_content.state, dict)
                and _UMS_CONVERSATION_ID in custom_content.state
            ):
                return custom_content.state.get(_UMS_CONVERSATION_ID)
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""

        url = f"{self.ums_agent_endpoint.rstrip('/')}/conversations"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={})
            response.raise_for_status()
            data = response.json()
        return data.get("id")

    async def __call_ums_agent(
        self, conversation_id: str, user_message: str, stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""

        url = f"{self.ums_agent_endpoint.rstrip('/')}/conversations/{conversation_id}/chat"
        payload = {
            "message": {
                "role": "user",
                "content": user_message,
            },
            "stream": True,
        }

        content = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        stage.append_content(chunk)
                        content += chunk

        return content
