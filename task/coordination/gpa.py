from copy import deepcopy
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    CustomContent,
    Message,
    Request,
    Role,
    Stage,
)
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        """Call GPA, stream output, and propagate stages/attachments/state."""

        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version="2025-01-01-preview",
        )

        extra_headers = {}
        conversation_id = request.headers.get("x-conversation-id")
        if conversation_id:
            extra_headers["x-conversation-id"] = conversation_id

        chunks = await client.chat.completions.create(
            messages=self.__prepare_gpa_messages(request, additional_instructions),
            stream=True,
            deployment_name="general-purpose-agent",
            extra_headers=extra_headers or None,
        )

        content = ""
        result_custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        async for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.content:
                print(delta.content, end="")
                stage.append_content(delta.content)
                content += delta.content

            if delta.custom_content:
                custom_content = delta.custom_content
                if custom_content.attachments:
                    result_custom_content.attachments.extend(custom_content.attachments)

                if custom_content.state:
                    if result_custom_content.state is None:
                        result_custom_content.state = {}
                    if isinstance(custom_content.state, dict):
                        result_custom_content.state.update(custom_content.state)

                custom_content_dict = custom_content.dict(exclude_none=True)
                stages_data = custom_content_dict.get("stages")
                if stages_data:
                    for stg in stages_data:
                        stg_index = stg.get("index")
                        if stg_index is None:
                            continue

                        stg_obj = stages_map.get(stg_index)
                        if not stg_obj:
                            stg_obj = StageProcessor.open_stage(choice, stg.get("name"))
                            stages_map[stg_index] = stg_obj

                        if stg.get("content"):
                            stg_obj.append_content(stg["content"])

                        attachments = stg.get("attachments") or []
                        for attachment in attachments:
                            stg_obj.add_attachment(
                                type=attachment.get("type"),
                                title=attachment.get("title"),
                                data=attachment.get("data"),
                                url=attachment.get("url"),
                                reference_url=attachment.get("reference_url"),
                                reference_type=attachment.get("reference_type"),
                            )

                        if stg.get("status") == "completed":
                            StageProcessor.close_stage_safely(stg_obj)

        if result_custom_content.attachments:
            for attachment in result_custom_content.attachments:
                if isinstance(attachment, Attachment):
                    choice.add_attachment(attachment)
                else:
                    choice.add_attachment(
                        Attachment(**attachment.dict(exclude_none=True))
                    )

        if result_custom_content.state:
            choice.set_state(
                {_IS_GPA: True, _GPA_MESSAGES: result_custom_content.state}
            )

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=result_custom_content,
        )

    def __prepare_gpa_messages(
        self, request: Request, additional_instructions: Optional[str]
    ) -> list[dict[str, Any]]:
        """Restore GPA history and append the latest user request."""

        res_messages: list[dict[str, Any]] = []

        for idx, message in enumerate(request.messages):
            if message.role == Role.ASSISTANT:
                custom_content = message.custom_content
                if custom_content and custom_content.state:
                    state = custom_content.state
                    if isinstance(state, dict) and state.get(_IS_GPA):
                        if idx > 0:
                            res_messages.append(
                                request.messages[idx - 1].dict(exclude_none=True)
                            )

                        restored_message = deepcopy(message)
                        restored_message.custom_content.state = state.get(_GPA_MESSAGES)
                        res_messages.append(restored_message.dict(exclude_none=True))

        last_message = deepcopy(request.messages[-1])
        if additional_instructions:
            last_message.content = (
                (last_message.content or "")
                + "\n\nAdditional instructions:\n"
                + additional_instructions
            )

        res_messages.append(last_message.dict(exclude_none=True))
        return res_messages
