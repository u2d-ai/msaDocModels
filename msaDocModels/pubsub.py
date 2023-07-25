from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, validator


class UpdateStatusTypes(Enum):
    """
    Enum Class that represents choices for statuses in pubsub message.
    """

    create = "create"
    update = "update"
    delete = "delete"


class DatabaseUpdateMessageDTO(BaseModel):
    """
    DTO that contains needed attributes to be processed.

    Attributes:
        class_name: Name of Model class.
        type: Type of update.
        entry: Data from database entry.
    """

    class_name: str
    type: str
    entry: Dict[str, Any]

    @validator("type")
    def validate_notification_type(cls, _type: str):
        available_types = [attr.value for attr in UpdateStatusTypes]
        if _type in available_types:
            return _type
        raise ValueError("Notification type must be a str type and be available in UpdateMessageTypes enum class")


class DatabaseUpdateInput(BaseModel):
    """
    Pydantic model to receive database update from pub/sub.

    Attributes:

        data: Database update.
    """

    data: DatabaseUpdateMessageDTO
