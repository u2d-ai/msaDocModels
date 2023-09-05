import datetime
from typing import TypeVar

from genson import SchemaBuilder
from pydantic.types import UUID

_T = TypeVar("_T")

try:
    import orjson as json
except:
    try:
        import ujson as json

    except:
        import json

        json.__version__ = ""


import uuid
from typing import List, Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


class titem(BaseModel):
    item_me: str = "Stefan"


class test(BaseModel):
    info: str = "some information"
    test_list: List[str] = ["a", "list"]
    test_item: titem = titem()
    test_item_list: List[titem] = [titem()]
    test_optional_item: Optional[titem] = None
    test_date: datetime.datetime = datetime.datetime.utcnow()


class Metadata(BaseModel):
    msg_type: Optional[str] = None
    msg_id: UUID = uuid.uuid4()
    sender_id: UUID = uuid.uuid4()
    created_at: datetime.datetime = datetime.datetime.utcnow()
    json_module: str = json.__name__
    json_version: str = json.__version__
    metaclass: str = ""


class MSAAPIMessage:
    def __init__(self, metaClass: _T = Metadata, **kwargs):
        self.__dict__ = kwargs
        self.metadata: metaClass = metaClass()
        self.metadata.metaclass = metaClass.__name__

    def toDict(self) -> dict:
        return jsonable_encoder(self.__dict__)

    def fromDict(self, msg: dict):
        self.__dict__ = msg

    def toMsg(self) -> str:
        classes = {}
        for k, v in self.__dict__.items():
            classes[k] = v.__class__.__name__
            # print("TOMSG:",k,v, type(v), v.__class__.__name__)
        self.__dict__["classes"] = classes
        return json.dumps(self, default=lambda x: jsonable_encoder(x.__dict__)).decode("utf8").replace("'", '"')

    def fromMsg(self, message: str):
        self.__dict__ = json.loads(message)

    def fromMsgToClasses(self, message: str):
        self.fromMsg(message=message)
        if "metaclass" in self.metadata.keys():
            mc = globals()[self.metadata["metaclass"]]
            self.metadata = mc(**self.__dict__.pop("metadata"))
        if "classes" in self.__dict__.keys():
            for k, v in self.__dict__["classes"].items():
                try:
                    mv = globals()[v]
                    self.__dict__[k] = mv(**self.__dict__[k])
                except:
                    if not isinstance(self.__dict__[k], dict):
                        mv = v
                        try:
                            self.__dict__[k] = type(mv)(self.__dict__[k])
                        except:
                            pass

    def schema_json(self) -> str:
        builder = SchemaBuilder()
        builder.add_object(self.toDict())
        return builder.to_json()


if __name__ == "__main__":
    m = MSAAPIMessage(
        info="some information",
        mylist=["a", "list"],
        test=test(),
    )

    print("m todict:", m.toDict())
    print("m tomsg:", m.toMsg())
    print("m str:", m.__str__())
    print("m api:", jsonable_encoder(m))
    print("m rep:", m.__repr__())
    print("m cls:", m.__class__)
    print("schema:", m.schema_json())

    print("old instance test", m.test)
    print("old instance test", type(m.test))

    mya = MSAAPIMessage()
    print("New instance after blank initialize", mya.toMsg())
    mya.fromMsg(m.toMsg())
    print("New instance after fromMsg", mya.toMsg())
    print("New instance meta", mya.metadata)
    print("New instance type of mya.test", type(mya.test))

    mya.fromMsgToClasses(m.toMsg())
    print("New instance after fromMsgToClasses", mya.toMsg())
    print("New instance meta", mya.metadata)
    print("New instance type of mya.test", type(mya.test))
