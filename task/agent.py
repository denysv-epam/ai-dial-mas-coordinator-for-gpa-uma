import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Choice, Message, Request, Role, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import AgentName, CoordinationRequest
from task.prompts import (
    COORDINATION_REQUEST_SYSTEM_PROMPT,
    FINAL_RESPONSE_SYSTEM_PROMPT,
)
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        """Coordinate routing, run the chosen agent, and stream final output."""

        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version="2025-01-01-preview",
        )

        coordination_stage = StageProcessor.open_stage(choice, "Coordination Request")
        coordination_request = await self.__prepare_coordination_request(
            client, request
        )
        coordination_stage.append_content(
            json.dumps(coordination_request.dict(exclude_none=True), indent=2)
        )
        StageProcessor.close_stage_safely(coordination_stage)

        agent_stage = StageProcessor.open_stage(
            choice, f"{coordination_request.agent_name.value} Agent"
        )
        try:
            agent_message = await self.__handle_coordination_request(
                coordination_request=coordination_request,
                choice=choice,
                stage=agent_stage,
                request=request,
            )
        finally:
            StageProcessor.close_stage_safely(agent_stage)

        return await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_message=agent_message,
        )

    async def __prepare_coordination_request(
        self, client: AsyncDial, request: Request
    ) -> CoordinationRequest:
        """Ask the LLM to select an agent and parse the JSON response."""

        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema(),
                    },
                }
            },
        )

        content = None
        if response.choices:
            content = response.choices[0].message.content

        if isinstance(content, dict):
            data = content
        else:
            data = json.loads(content or "{}")

        try:
            return CoordinationRequest.model_validate(data)
        except Exception as exc:
            logger.warning(
                "Failed to parse coordination request, defaulting to GPA", exc_info=exc
            )
            return CoordinationRequest(
                agent_name=AgentName.GPA, additional_instructions=None
            )

    def __prepare_messages(
        self, request: Request, system_prompt: str
    ) -> list[dict[str, Any]]:
        """Convert DIAL messages to OpenAI-style payload with a system prompt."""

        messages: list[dict[str, Any]] = [
            {
                "role": Role.SYSTEM.value,
                "content": system_prompt,
            }
        ]

        for message in request.messages:
            role_value = (
                message.role.value if hasattr(message.role, "value") else message.role
            )
            if role_value == Role.USER.value and message.custom_content:
                messages.append(
                    {
                        "role": Role.USER.value,
                        "content": message.content or "",
                    }
                )
            else:
                msg_dict = message.dict(exclude_none=True)
                role = msg_dict.get("role")
                if isinstance(role, Role):
                    msg_dict["role"] = role.value
                messages.append(msg_dict)

        return messages

    async def __handle_coordination_request(
        self,
        coordination_request: CoordinationRequest,
        choice: Choice,
        stage: Stage,
        request: Request,
    ) -> Message:
        """Dispatch to UMS or GPA gateway based on the coordination request."""

        if coordination_request.agent_name == AgentName.UMS:
            ums_gateway = UMSAgentGateway(self.ums_agent_endpoint)
            return await ums_gateway.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )

        if coordination_request.agent_name == AgentName.GPA:
            gpa_gateway = GPAGateway(self.endpoint)
            return await gpa_gateway.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )

        raise ValueError(f"Unsupported agent name: {coordination_request.agent_name}")

    async def __final_response(
        self,
        client: AsyncDial,
        choice: Choice,
        request: Request,
        agent_message: Message,
    ) -> Message:
        """Synthesize and stream the final response using agent context."""

        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        user_request = request.messages[-1].content or ""
        agent_context = agent_message.content or ""
        augmented_prompt = (
            "Context from called agent:\n"
            f"{agent_context}\n\n"
            "User request:\n"
            f"{user_request}"
        )
        if messages:
            messages[-1]["content"] = augmented_prompt

        chunks = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            stream=True,
        )

        content = ""
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )
