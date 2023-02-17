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
    """
    Input model with tenant_id

    Attributes:
        tenant_id: tenant's uuid.
    """

    tenant_id: UUID4


class TextInput(BaseModel):
    """
    Input model with input_text

    Attributes:
        input_text: input text.
    """

    input_text: Union[str, List, Dict]


class DocumentInput(TextInput):
    """
    Input document model

    Attributes:
        document_id: optional uuid for document.
    """

    document_id: Optional[UUID4]


class SentencesInput(BaseModel):
    """
    Input model with sentences

    Attributes:
        document_id: optional uuid for document.
        sentences: list of sentences
    """

    document_id: Optional[UUID4]
    sentences: List[str]


class DocumentIds(BaseModel):
    """
    Ids of documents from mail model

    Attributes:
        document_ids: list of uuids.
    """

    document_ids: List[str]


class DocumentLangInput(DocumentInput):
    """
    Input document model made over SDULanguage. Default language ENGLISH

    Attributes:
        language: object SDULanguage.
    """

    language: SDULanguage = SDULanguage(code="en", lang="ENGLISH")


class SPKLanguageInput(DocumentInput):
    """
    Input model to detect language.

    Attributes:
        hint_languages: language hint for analyzer. 'ITALIAN' or 'it' boosts Italian;see LANGUAGES for known languages.
        hint_encoding: encoding hint for analyzer. 'SJS' boosts Japanese; see ENCODINGS for all known encodings.
        sentence_detection: turn on/off language detection by sentence.
        get_vectors: turn on/off return of sentence vectors.
        is_plain_text: if turned off, and HTML is provided as input, detection will skip HTML tags,
            expand HTML entities and detect HTML <lang ...> tags.
        is_short_text: turn on to get the best effort results (instead of unknown) for short text.
    """

    hint_languages: str = ""
    hint_encoding: str = ""
    sentence_detection: bool = True
    get_vectors: bool = True
    is_plain_text: bool = True
    is_short_text: bool = False


class SPKLanguageDTO(sdu.SDULanguageDetails):
    """DTO, representing the result of service language."""


class TextWithParagraphsGet(BaseModel):
    """
    Schema representing the result of paragraph segmentation.

    Attributes:
        paragraphs: list of SDUParagraph.
    """

    paragraphs: List[sdu.SDUParagraph]


class TextWithSentencesGet(BaseModel):
    """
    Schema representing the result of sentences segmentation.

    Attributes:
        sentences: list of SDUSentence.
    """

    sentences: List[sdu.SDUSentence]


class TextWithPagesGet(BaseModel):
    """
    Schema representing the result of pages segmentation.

    Attributes:
        pages: list of SDUPage.
    """

    pages: List[sdu.SDUPage]


class SPKSegmentationInput(BaseModel):
    """
    Input model to detect Segmentation

    Attributes:
        document_id: optional uuid for document.
        input_text: input_text.
        language: SDULanguage object for this text.

    """

    document_id: Optional[UUID4]
    input_text: Union[str, List[str], Dict[int, str]]
    language: SDULanguage = SDULanguage(code="en", lang="ENGLISH")


class SPKSegmentationDTO(BaseModel):
    """
    DTO, representing the result of service segmentation. Only one attribute will be non-empty.

    Attributes:
        pages: list of SDUPage.
        paragraphs: list of SDUParagraph.
        sentences: list of SDUSentence.
    """

    pages: List[sdu.SDUPage] = []
    paragraphs: List[sdu.SDUParagraph] = []
    sentences: List[sdu.SDUSentence] = []


class SPKTextCleanInput(DocumentInput):
    """Data input model for Text Clean."""


class SPKTextCleanDTO(BaseModel):
    """
    DTO, representing the result of service text clean.

    Attributes:
        text: cleaned text.
    """

    text: str


class SPKDataCleanAIInput(BaseModel):
    """
    Data input model for Text AI Clean.

    Attributes:

        text: List of dictionaries
        keys: The keys  which need to clean
    """

    text: List[Dict[str, Dict[str, Any]]]
    keys: List[str] = []


class SPKDataCleanAIDTO(BaseModel):
    """
    DTO, representing the result of service ai text clean.

    Attributes:

        text: List of dictionaries
    """

    data: List[Dict[str, Dict[str, Any]]]


class SPKSentimentInput(DocumentInput):
    """Data input model for Sentiment."""


class SPKSentimentDTO(BaseModel):
    """
    DTO, representing the result of service Sentiment.

    Attributes:
        neg:Negativity score of the text.
        neu: Neutrality score of the text.
        pos: Positivity score of the text.
        compound: Compound score of the text.
        error: None if there is no errors, otherwise contains description of the error.
    """

    neg: Optional[float]
    neu: Optional[float]
    pos: Optional[float]
    compound: Optional[float]
    error: Optional[str]


class SPKPhraseMiningInput(DocumentLangInput):
    """Data input model for Phrase mining."""


class SPKPhraseMiningDTO(BaseModel):
    """
    DTO, representing the result of Phrase mining.

    Attributes:
        phrases: Nested list of most common phrases in the provided sentence(s)
    """

    phrases: List[Union[List, List[Union[str, int]]]]


class SPKWeightedKeywordsDTO(BaseModel):
    """
    DTO, representing the result of service Keywords.

    Attributes:
        keywords:  List of keywords and/or keyphrases.
    """

    keywords: List[Union[List, List[Union[str, int]]]]


class SPKExtractKeywordsInput(BaseModel):
    """
    Data input model for ExtractKeywords.

    Attributes:
        data: extended input text by InputKeyKeys, have the len as input.
        algorithms: which algorithms use for extract. Can be list of ["yake", "bert", "bert_vectorized", "tf_idf]
        keys: which keys need to extract
    """

    data: List[Dict[str, Dict[str, Any]]]
    algorithms: List[str] = ["yake", "bert"]
    keys: List[str] = []


class SPKExtractKeywordsDTO(BaseModel):
    """
    DTO, representing the result of service Keywords.

    Attributes:
        data: extended input text by InputKeyKeys, have the len as input.
    """

    data: List[Dict[str, Dict[str, Any]]]


class SPKSummaryInput(DocumentLangInput):
    """
    Data input model for Summary.

    Attributes:
        sum_ratio: Coefficient.
        sentences_count: Amount of sentences.
        lsa: Algorithm
        corpus_size: Coefficient
        community_size: Coefficient
        cluster_threshold: Coefficient
    """

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
    """
    Data input model for Notary.

    Attributes:
        city: default city to check.
    """

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
    """
    DTO, representing the result of service Taxonomy Cities.

    Attributes:

        cities: List of SPKCities.
        cities_winner: winner object SPKCity.
    """

    cities: List[SPKCity]
    cities_winner: Optional[SPKCity]


class SPKTaxonomyCountriesDTO(BaseModel):
    """
    DTO, representing the result of service Taxonomy Countries.

    Attributes:

        countries: List of SPKCountries.
        countries_winner: winner object SPKCountry.
    """
    countries: List[SPKCountry]
    countries_winner: Optional[SPKCountry]


class SPKTaxonomyCompaniesDTO(BaseModel):
    """
    DTO, representing the result of service Taxonomy Companies.

    Attributes:

        companies: List of SPKCompanies.
        companies_winner: winner object SPKCompany.
    """

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
        vars: Vars to provide another settings.
        correlations: Settings regarding correlation metrics and thresholds.
        sort: Default sorting.
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
    vars: Dict = {}
    data: List[Dict[str, Any]]
    sort: str = "ascending"
    progress_bar: bool = False
    minimal: bool = False
    explorative: bool = False
    sensitive: bool = False
    dark_mode: bool = False
    orange_mode: bool = False


class SPKProfileDTO(BaseModel):
    """
    Pydantic model of Profile HTML representation
    """


class SPKBuildModelInput(BaseModel):
    """
    Model that contains input data for building a machine learning model.

    Attributes:

        model_name: The name of the model.
        data: The input data to be used for model training.
        target_columns: Column names representing the target variable(s) to be predicted.
        train_columns: Column names to be used for model training.
        text_features: Column names representing text features to be used for text processing.
        ignore_features: Column names representing features to be ignored during model training.
        categorical_features: Column names representing categorical features.
        date_features: Column names representing date features.
        numeric_features: Column names representing numeric features.
        ordinal_features: Dictionary of column names representing ordinal features and their categories.
        multiplier: Multiplier used for increasing the size of the training data using synthetic samples.
        session_id: Seed value used for reproducibility.
        remove_outliers: Flag indicating whether to remove outliers from the data.
        budget_time_minutes: Maximum time in minutes allowed for model training.
        included_engines: List of machine learning models to be used for model training.
        use_gpu: Flag indicating whether to use GPU for model training.
        fold: Number of folds for cross-validation.
        tuning_iterations: Number of iterations for hyperparameter tuning.
        create_metadata: Flag indicating whether to create model metadata
    """
    model_name: str = "kim_pipeline"
    data: List[Dict[str, Any]]
    target_columns: List[str] = []
    train_columns: List[str] = []
    text_features: List[str] = []
    ignore_features: List[str] = []
    categorical_features: List[str] = []
    date_features: List[str] = []
    numeric_features: List[str] = []
    ordinal_features: dict[str, list] = {}
    multiplier: int = 5
    session_id: int = 123
    remove_outliers: bool = False
    budget_time_minutes: float = 3.0
    included_engines: List[str] = ["svm", "nb", "ridge", "rf", "dt"]
    use_gpu = False
    fold: int = 7
    tuning_iterations: int = 7
    create_metadata = False


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

    Attributes:

        id: The id of element in mongo.
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


class SPKConversionInput(BaseModel):
    """
    Model that contains inference data along with filenames to use for XLSX conversion.

    Attributes:

        filenames: list of filenames that files should be saved as
        inference: inference data, first key means sheet name for XLSX file
    """

    filenames: List[str]
    inference: List[Dict[str, Dict[str, Any]]]


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
    """
    Matching pydantic models with fields in the db.

    Attributes:
        content_attachments: list of SDUAttachments.
        txt_content: SDUText.
        msg: SDUEmail.
        content_unzipped_files: object of
    """

    content_attachments: List[SDUAttachment]
    txt_content: SDUText
    msg: SDUEmail
    content_unzipped_files: Optional[List[SPKHTMLConverterResponse]]


class FieldName(str, Enum):
    """
    Matching pydantic models with fields in the db.

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
