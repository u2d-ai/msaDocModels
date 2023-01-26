import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from bson.objectid import ObjectId
from pydantic import UUID4, BaseModel, Field

from msaDocModels import sdu, wdc
from msaDocModels.sdu import (
    SDUAttachment,
    SDUContent,
    SDUData,
    SDUEmail,
    SDULanguage,
    SDUText,
)


class TenantIdInput(BaseModel):
    """Input model with tenant_id"""

    tenant_id: UUID4


class TextInput(BaseModel):
    """Input model with input_text"""

    input_text: Union[str, List, Dict]


class DocumentInput(TextInput):
    """Input document model"""

    document_id: Optional[UUID4]


class SentencesInput(BaseModel):
    """Input model with sentences"""

    document_id: Optional[UUID4]
    sentences: List[str]


class DocumentIds(BaseModel):
    """Ids of documents from mail model"""

    document_ids: List[str]


class DocumentLangInput(DocumentInput):
    """Input document model made over SDULanguage. Default language ENGLISH"""

    language: SDULanguage = SDULanguage(code="en", lang="ENGLISH")


class SPKLanguageInput(DocumentInput):
    """Input model to detect language."""

    hint_languages: str = ""
    hint_encoding: str = ""
    sentence_detection: bool = True
    get_vectors: bool = True
    is_plain_text: bool = True
    is_short_text: bool = False


class SPKLanguageDTO(sdu.SDULanguageDetails):
    """DTO, representing the result of service language."""


class TextWithParagraphsGet(BaseModel):
    """Schema representing the result of paragraph segmentation."""

    paragraphs: List[sdu.SDUParagraph]


class TextWithSentencesGet(BaseModel):
    """Schema representing the result of sentences segmentation."""

    sentences: List[sdu.SDUSentence]


class TextWithPagesGet(BaseModel):
    """Schema representing the result of pages segmentation."""

    pages: List[sdu.SDUPage]


class SPKSegmentationInput(BaseModel):
    """Input model to detect Segmentation"""

    document_id: Optional[UUID4]
    input_text: Union[str, List[str], Dict[int, str]]
    language: SDULanguage = SDULanguage(code="en", lang="ENGLISH")


class SPKSegmentationDTO(BaseModel):
    """DTO, representing the result of service segmentation. Only one attribute will be non-empty."""

    pages: List[sdu.SDUPage] = []
    paragraphs: List[sdu.SDUParagraph] = []
    sentences: List[sdu.SDUSentence] = []


class SPKTextCleanInput(DocumentInput):
    """Data input model for Text Clean."""


class SPKTextCleanDTO(BaseModel):
    """DTO, representing the result of service text clean."""

    text: str


class SPKSentimentInput(DocumentInput):
    """Data input model for Sentiment."""


class SPKSentimentDTO(BaseModel):
    """DTO, representing the result of service Sentiment."""

    neg: Optional[float]
    neu: Optional[float]
    pos: Optional[float]
    compound: Optional[float]
    error: Optional[str]


class SPKPhraseMiningInput(DocumentLangInput):
    """Data input model for Phrase mining."""


class SPKPhraseMiningDTO(BaseModel):
    """DTO, representing the result of Phrase mining."""

    phrases: List[Union[List, List[Union[str, int]]]]


class SPKWeightedKeywordsDTO(BaseModel):
    """DTO, representing the result of service Keywords."""

    keywords: List[Union[List, List[Union[str, int]]]]


class SPKSummaryInput(DocumentLangInput):
    """Data input model for Summary."""

    sum_ratio: float = 0.2
    sentences_count: int = 15
    lsa: bool = False
    corpus_size: int = 5000
    community_size: int = 5
    cluster_threshold: float = 0.65


class SPKStatisticsInput(DocumentLangInput):
    """Data input model for Statistics."""


class SPKStatisticsDTO(sdu.SDUStatistic):
    """DTO, representing the result of service Statistics."""


class SPKSummaryDTO(wdc.WDCItem):
    """DTO, representing the result of service Summary."""


class SPKNotaryInput(DocumentInput):
    """Data input model for Notary."""

    city: str = "Bremen"


class SPKNotary(BaseModel):
    """Detected Notary Pydantic Model."""

    sid: Optional[str]
    last_name: Optional[str]
    first_name: Optional[str]
    zip_code: Optional[str]
    city: Optional[str]
    office_city: Optional[str]
    official_location: Optional[str]
    address: Optional[str]
    additional_address: Optional[str]
    title: Optional[str]
    phone: Optional[str]
    complete_name_with_official_location: Optional[str]
    local_city: str = "Bremen"
    is_local_city: bool


class SPKNotaryWinnerDTO(SPKNotary):
    """DTO, representing the result of service Notary."""


class SPKCountry(BaseModel):
    """Detected Country Pydantic Model."""

    name: str
    official: str
    currencies: Dict[str, Dict[str, str]]
    capital: List[str]
    region: str
    subregion: str
    languages: Dict[str, str]
    latlng: List[int]
    flag: str
    calling_codes: List[str]


class SPKCompany(BaseModel):
    """Detected Company Pydantic Model."""

    rank: int
    company: str
    employees: str
    change_in_rank: str
    industry: str
    description: str
    revenue: str
    revenue_change: str
    profits: str
    profit_change: str
    assets: str
    market_value: str


class SPKCity(BaseModel):
    """Detected City Pydantic Model."""

    name: str
    country: str
    latlng: List[float]


class SPKTaxonomyCitiesDTO(BaseModel):
    """DTO, representing the result of service Taxonomy Cities."""

    cities: List[SPKCity]
    cities_winner: Optional[SPKCity]


class SPKTaxonomyCountriesDTO(BaseModel):
    """DTO, representing the result of service Taxonomy Countries."""

    countries: List[SPKCountry]
    countries_winner: Optional[SPKCountry]


class SPKTaxonomyCompaniesDTO(BaseModel):
    """DTO, representing the result of service Taxonomy Companies."""

    companies: List[SPKCompany]
    companies_winner: Optional[SPKCompany]


class SPKTaxonomyDTO(
    SPKTaxonomyCountriesDTO, SPKTaxonomyCompaniesDTO, SPKTaxonomyCitiesDTO
):
    """DTO, representing the result of service Taxonomy."""


class SPKTaxonomyInput(DocumentInput):
    """Data input model for Taxonomy."""


class AutoMLStatus(BaseModel):
    """
    Pydantic model to receive/send service status for pub/sub.

    Attributes:

        info: Service status.
        id: UUID model identifier.
        path: The path where model is located
    """

    info: str
    id: Optional[uuid.UUID]
    path: Optional[str]


class SPKProfileInput(BaseModel):
    """
    Pydantic model to generate a profile report based on data

    Attributes:
        title: Title of HTML representation.
        data: List of data.
        missing_diagrams: Settings related with the missing data section and the visualizations it can include.
        correlations: Settings regarding correlation metrics and thresholds.
        progress_bar: If True will display a progress bar.
        minimal: Minimal mode is a default configuration with minimal computation.
        explorative: Explorative mode.
        sensitive: Sensitive mode.
        dark_mode: Select a dar theme.
        orange_mode: Select a orange theme.

    """

    title: str
    html: Dict = {}
    missing_diagrams: Dict = {}
    correlations: Dict = {}
    data: List[Dict[str, Any]]
    progress_bar: bool = False
    minimal: bool = False
    explorative: bool = False
    sensitive: bool = False
    dark_mode: bool = False
    orange_mode: bool = False


class SPKProfileDTO(BaseModel):
    """
    Pydantic model of Profile HTML representation

    Attributes:

        data: Profile html representation.
    """

    data: str


class SPKLearnsetInput(BaseModel):
    """
    Pydantic model of Profile HTML representation AI Prediction input

    Attributes:
        name: Name of model.
        data: List of data.
        target_fields: Name of the target column in data.
        train_fields: List of column names that contain a text corpus.
        ml_n_models: Number of training models.
        optimize: Metric to use for model selection.
        algorithm: ID of an estimator available in model library.
    """

    name: str
    data: List[Dict[str, Any]]
    target_fields: str
    train_fields: List[str]
    ml_n_models: int = 3
    optimize: str = "Recall"
    algorithm: str = "svm"


class SPKInferenceInput(BaseModel):
    """
     Pydantic model for get inference data.

    Attributes:

        path: The path where model is located.
        data: Profile html representation.
    """

    path: str
    data: List[Dict[str, Any]]


class SPKInferenceDTO(BaseModel):
    """
    Pydantic model, provided merged inference data.

    Attributes:

        inference: Raw data with inference data.
    """

    inference: List[Dict[str, Any]]


class ProcessStatus(BaseModel):
    """
    Workflow status

        Attributes:

        number: number of status
        timestamp: time when number was changes
    """

    number: int = 0
    timestamp: str = str(datetime.utcnow())


class SPKDBBaseDocumentInput(BaseModel):
    """
    Document fields for input.

    Attributes:

        uid: document uid
        name: document name.
        mimetype: mimetype.
        path: path to file.
        layout_file_path: path to layout file.
        debug_file_path: path to debug file.
        readorder_file_path: path to rearorder file.
        folder: folder name.
        group_uuid: group identifier.
        size_bytes: size in bytes.
        is_file: file or not.
        wfl_status: wfl status.
        import_status: import status.
        user: user name.
        date: date.
        runtime_s: runtime in sec.
        tags: list of tags.
        language: language.
        needs_update: need update or not.
        data: data.
        project_code: project code.
        npages: count of pages.
        content: content.
        metadata: metadata.
        description: discription.
        status: document status
        text: text.
        file: file.
        sdu: Dict of sdu objects.
    """

    uid: str
    name: str
    mimetype: str = "text/plain"
    path: str = ""
    layout_file_path: str = ""
    debug_file_path: str = ""
    readorder_file_path: str = ""
    folder: str = ""
    group_uuid: str = ""
    size_bytes: int = 0
    is_file: bool = False
    wfl_status: List = []
    import_status: str = "new"
    user: str = ""
    date: str = ""
    runtime_s: float = 0.0
    tags: Optional[Dict] = {}
    language: Optional[SDULanguage] = None
    needs_update: bool = False
    data: Optional[SDUData] = None
    project_code: str = ""
    npages: int = 0
    content: Optional[SDUContent] = None
    metadata: Dict = {}
    description: str = ""
    text: str = ""
    file: Dict = {}
    sdu: Dict = {}
    status: ProcessStatus = ProcessStatus()
    status_history: List[ProcessStatus] = [ProcessStatus()]


class PyObjectId(ObjectId):
    """
    Converts ObjectId to string.
    """

    @classmethod
    def __get_validators__(cls):
        """
        Generator to return validate method.
        """
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """
        Validates Object ID.

        Parameters:

             v: value to validate.

        Returns:

            Object ID with specified value.

        Raises:

            ValueError if Object ID does not pass validation.
        """
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class MongoId(BaseModel):
    """
    MongoDB _id field.
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}


class SPKDBBaseDocumentDTO(SPKDBBaseDocumentInput, MongoId):
    """
    Document fields for output.
    """


class BaseInfo(BaseModel):
    """
    Base info for AI stuff.

    Attributes:

        version: version identifier.
        description: description.
        datetime: datetime.
        inherited: inherited or not.
        active: active or not.
        name: object name.
    """

    version: str
    description: str
    datetime: datetime
    inherited: bool
    active: bool
    name: str


class SPKUpdateAI(BaseModel):
    """
    Update ai fields.

    Attributes:

        version: version identifier.
        description: description.
        datetime: datetime.
        inherited: inherited or not.
        active: active or not.
        name: object name.
    """

    version: Optional[str]
    description: Optional[str]
    datetime: Optional[datetime]
    inherited: Optional[bool]
    active: Optional[bool]
    name: Optional[str]


class SPKLearnsetDataInput(BaseInfo):
    """
    AI learnset input.

    Attributes:

        learnsets: list of learnset objects.
    """

    learnsets: List[Dict]


class SPKTestsetDataInput(BaseInfo):
    """
    AI testset input.

    Attributes:

        testsets: list of testsets objects.
    """

    testsets: List[Dict]


class SPKTaxonomyDataInput(BaseInfo):
    """
    AI taxonomy input.

    Attributes:

        taxonomies: list of taxonomies objects.
    """

    taxonomies: List[Dict]


class SPKModelDataInput(BaseInfo):
    """
    AI model input.

    Attributes:

        model: model object.
    """

    model: Dict


class SPKTestsetDataDTO(SPKTestsetDataInput, MongoId):
    """
    AI testset output.
    """


class SPKLearnsetDataDTO(SPKLearnsetDataInput, MongoId):
    """
    AI learnset output.
    """


class SPKModelDataDTO(SPKModelDataInput, MongoId):
    """
    AI model output.
    """


class SPKTaxonomyDataDTO(SPKTaxonomyDataInput, MongoId):
    """
    AI taxonomy output.
    """


class SPKHTMLConverterResponse(BaseModel):
    """
    Response from converter

    Attributes:

        metadata: metadata from file
        txt_content: SDUText object
    """

    metadata: Dict
    txt_content: SDUText


class SPKEmailConverterResponse(BaseModel):
    content_attachments: List[SDUAttachment]
    txt_content: SDUText
    msg: SDUEmail
    content_unzipped_files: Optional[List[SPKHTMLConverterResponse]]


class FieldName(str, Enum):
    """Matching pydantic models with fields in the db.
    Attributes:
        TestsetDataInput: name of testset input model.
        LearnsetDataInput: name of learnset input model.
        ModelDataInput: name of model input model.
        TaxonomyDataInput: name of taxonomy input model.
    """

    TestsetDataInput = "testset"
    LearnsetDataInput = "learnset"
    ModelDataInput = "model"
    TaxonomyDataInput = "taxonomy"
