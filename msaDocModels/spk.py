import html
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from bson.objectid import ObjectId
from pydantic import UUID4, BaseModel, Field

from msaDocModels import wdc
from msaDocModels.utils.htmlutils import sanitize


def get_crlf() -> str:
    """get's the OS Environment Variable for ``CR_LF``.
    Default: ``\\n``
    """
    ret: str = os.getenv("CR_LF", "\n")
    return ret


def get_sentence_seperator() -> str:
    """get's the OS Environment Variable for ``SENTENCE_SEPARATOR``.
    Default: `` `` (Space/Blank)
    """
    ret: str = os.getenv("SENTENCE_SEPARATOR", " ")
    return ret


def get_cr_paragraph() -> str:
    # CR_PARAGRAPH
    """get's the OS Environment Variable for ``CR_PARAGRAPH``.
    Default: ``\\n\\n``
    """
    ret: str = os.getenv("CR_PARAGRAPH", "\n\n")
    return ret


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


class SDUPageImage(BaseModel):
    """Page Image Pydantic Model.

    Storing the information about the Image representation of a Page.
    """

    id: int = -1  # ID = Page Index.
    filepath_name: str = ""  # Filepath to the image on filesystem storage.
    height: float = 0.0  # Image Height.
    width: float = 0.0  # Image Width.
    dpi: float = 0.0  # Picture DPI Resolution.
    format: str = ""  # Image Format (png, jpg etc.).
    mode: str = ""  # Image Mode.
    layout: List = []  # Image Layout Information."""

    class Config:
        orm_mode = False


class SDUEmail(BaseModel):
    """Parsed EMail Pydantic Model."""

    msg_id: str = ""
    msg_from: str = ""
    msg_to: str = ""
    msg_cc: str = ""
    msg_bcc: str = ""
    msg_subject: str = ""
    msg_sent_date: str = ""
    msg_body: str = ""
    seg_body: str = ""  # Segmented Body (Signature, etc.)
    seg_sign: str = ""
    msg_sender_ip: str = ""
    msg_to_domains: str = ""
    msg_received: List = []
    msg_reply_to: str = ""
    msg_timezone: str = ""
    msg_headers: Dict = {}

    class Config:
        orm_mode = False


class SDUDetailLanguage(BaseModel):
    """Detailed Language Pydantic Model."""

    multiple: bool = False
    reliable: bool = False
    bytes: int = -1
    details: Optional[Tuple] = tuple()
    vectors: Optional[Tuple] = tuple()

    class Config:
        orm_mode = False


class SDULanguage(BaseModel):
    """Detected Language Pydantic Model."""

    code: str = "unknown"  # Short de, en etc.
    lang: str = "unknown"  # Language name like german.
    reliable: bool = False  # is the detected result reliable.
    proportion: int = -1  # Proportion of the text in this language.
    bytes: int = -1  # Bytes of the text in this language.
    confidence: float = -1  # Confidence from 0.01 to 1.0.
    winner: Optional[str] = None  # Selected overall Winner

    class Config:
        orm_mode = False


class SDULanguageDetails(SDULanguage):
    details: Optional[List] = list()  # Details of the top 3 detected languages.


class SDUStatistic(BaseModel):
    """Text Statistics Pydantic Model."""

    avg_character_per_word: float = 0
    avg_letter_per_word: float = 0
    avg_sentence_length: float = 0
    avg_syllables_per_word: float = 0
    avg_sentence_per_word: float = 0
    difficult_words: int = 0
    lexicon_count: int = 0
    long_word_count: int = 0
    reading_time_s: float = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    reading_ease_score: float = 0
    reading_ease: str = ""
    grade: float = 0
    smog: float = 0
    coleman: float = 0
    reading_index: float = 0
    reading_score: float = 0
    write_formula: float = 0
    fog: float = 0
    standard: str = ""
    crawford: float = 0
    gulpease_index: float = 0
    osman: float = 0

    class Config:
        orm_mode = False


class SDUSentence(BaseModel):
    id: int = -1
    text: str = ""
    xpos: List[str] = []
    upos: List[str] = []
    tokens: List[str] = []
    text_defaults: List = []
    text_ner: List = []
    text_nlp: List = []
    text_ml: List = []
    text_wsd: List = []

    class Config:
        orm_mode = False


class SDUPDFElement(BaseModel):
    line_id: int = -1
    span_id: int = -1
    flags: int = 0
    bold: bool = False
    italic: bool = False
    font: str = ""
    fontsize: float = 0.0
    color: int = 0


class SDUParagraph(BaseModel):
    id: int = -1
    sort: int = -1
    nsen: int = 0
    semantic_type: str = "text"
    section: str = "body"
    size_type: str = "body"
    sentences: List[SDUSentence] = []
    sentences_en: List[SDUSentence] = []
    clean: str = ""
    lang: SDULanguage = SDULanguage()
    elements: List[SDUPDFElement] = []

    class Config:
        orm_mode = False

    def has_text(self) -> bool:
        return len(self.sentences) > 0

    def get_text(self) -> str:
        return self.get_text_no_lf()

    def get_text_no_lf(self) -> str:
        ret = ""
        for sen in self.sentences:
            ret += sen.text + " "
        return ret

    def get_text_lf(self) -> str:
        ret = ""
        for sen in self.sentences:
            ret += sen.text + get_crlf()
        return ret


class SDUText(BaseModel):
    raw: str = ""
    clean: str = ""
    html_content: str = ""
    structured_content: Dict = {}
    lang: SDULanguage = SDULanguage()
    paragraphs: List[SDUParagraph] = []

    class Config:
        orm_mode = False


class SDUVersion(BaseModel):
    version: str = ""
    creation_date: str = ""

    class Config:
        orm_mode = False


class SDULearnset(BaseModel):
    # dict entry is the class, list are the train entrys for this class
    version: str = ""
    text: Dict = {}
    nlu: Dict = {}
    nlp: Dict = {}
    emb: Dict = {}
    vec_words: Dict = {}
    vec_sent: Dict = {}

    def set_version(self, version: str):
        self.version = version

    def reset(self):
        self.text.clear()
        self.nlu.clear()
        self.nlp.clear()
        self.emb.clear()
        self.vec_words.clear()
        self.vec_sent.clear()

    class Config:
        orm_mode = False


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


class SDUPage(BaseModel):
    id: int = -1
    npar: int = 0
    input: str = ""
    has_en: bool = False
    text: SDUText = SDUText()
    notary: SPKNotary = None

    class Config:
        orm_mode = False

    def has_text(self):
        return len(self.text.paragraphs) > 0

    def get_text_default(self):
        return self.get_text_no_lf()

    def get_text_no_lf(self):
        ret = ""
        for par in self.text.paragraphs:
            for sen in par.sentences:
                ret += sen.text + " "
        return ret

    def get_text_no_lf_en(self):
        ret = ""
        for par in self.text.paragraphs:
            for sen in par.sentences_en:
                ret += sen.text + " "
        return ret

    def get_text_no_lf_paragraph(self):
        ret = ""
        for par in self.text.paragraphs:
            for sen in par.sentences:
                ret += sen.text + get_cr_paragraph()
            ret += get_cr_paragraph()
        return ret

    def get_text_lf(self, space_before_lf: bool = False):
        ret = ""
        for par in self.text.paragraphs:
            for sen in par.sentences:
                ret += sen.text
                if space_before_lf:
                    ret += get_sentence_seperator()
                ret += get_crlf()
        return ret

    def get_text_lf_paragraph(self, space_before_lf: bool = False):
        ret = ""
        for par in self.text.paragraphs:
            for sen in par.sentences:
                ret += sen.text
                if space_before_lf:
                    ret += get_sentence_seperator()
                ret += get_sentence_seperator()
            ret += get_cr_paragraph()
        return ret

    def get_all_sentences_text_list_lf(self):
        ret = []
        for par in self.text.paragraphs:
            for i, sen in enumerate(par.sentences):
                txt = sen.text
                if i == len(par.sentences) - 1:
                    txt = txt + get_cr_paragraph()
                else:
                    txt = txt + get_crlf()
                ret.append(txt)
        return ret

    def get_text_for_nlp(self):
        ret = []

        for par in self.text.paragraphs:
            txt = ""
            for sen in par.sentences:
                txt += sen.text.replace("\n", "") + "\n\n"
            if len(txt) > 1:
                ret.append(txt)
        return ret

    def get_text_for_display(self):
        ret = ""
        for par in self.text.paragraphs:
            txt: str
            if par.semantic_type.__contains__("list"):
                txt = par.get_text()
                ret += txt + get_crlf()
            else:
                txt = par.get_text()
                if par.semantic_type.__contains__(
                    "head"
                ) or par.semantic_type.__contains__("title"):
                    ret += get_crlf() + txt + get_crlf()
                elif len(par.sentences) > 1:
                    ret += txt + get_cr_paragraph()
                else:
                    ret += txt + get_crlf()
        ret = ret.replace(get_cr_paragraph() + get_crlf(), get_cr_paragraph())
        return ret

    def get_all_sentences_text_list(self):
        ret = []

        for par in self.text.paragraphs:
            if par.semantic_type.__contains__("list"):
                txt = par.get_text_lf()
            else:
                txt = par.get_text()
            ret.append(txt)
        return ret

    def get_all_sentences_text_list_no_table_and_lists(self):
        ret = []

        for par in self.text.paragraphs:
            txt = ""
            if par.semantic_type.__contains__("table"):
                pass
            elif par.semantic_type.__contains__("list"):
                pass
            elif par.semantic_type.__contains__("image_text"):
                pass
            elif par.semantic_type.__contains__("imagetext"):
                pass
            elif par.semantic_type.__contains__("image"):
                pass
            elif par.semantic_type.__contains__("figure"):
                pass
            elif par.section.__contains__("footer"):
                pass
            elif par.semantic_type.__contains__("title"):
                pass
            elif par.semantic_type.__contains__("headline"):
                pass
            else:
                for i, sen in enumerate(par.sentences):
                    if len(txt) > 0:
                        txt += " "
                    txt = txt + sen.text  # getCRParagraph()
                ret.append(txt)
        return ret

    def get_all_sentences_text_list_en(self):
        ret = []
        for par in self.text.paragraphs:
            for i, sen in enumerate(par.sentences_en):
                txt = sen.text
                if i == len(par.sentences_en) - 1:
                    txt = txt + get_cr_paragraph()
                else:
                    txt = txt + get_sentence_seperator()
                ret.append(txt)
        return ret

    def set_input(self, input_text: str):
        self.input = input_text


class SDUData(BaseModel):
    npages: int = 0  #
    stats: SDUStatistic = SDUStatistic()  #
    pages: List[SDUPage] = []
    converter: List[str] = []
    email: SDUEmail = SDUEmail()  # email header
    text: SDUText = SDUText()  # parsed/body only
    images: List[SDUPageImage] = []

    class Config:
        orm_mode = False

    def add_page_pre_processing(self, page_pre: SDUPage):
        if page_pre:
            self.pages.append(page_pre)
            self.npages = len(self.pages)
            page_pre.page = self.npages

    def escaped(self):
        return html.escape(self.text.html_content)

    async def sanitized(self):
        return await sanitize(self.text.html_content)


class SDUBBox(BaseModel):
    x0: float = -1
    y0: float = -1
    x1: float = -1
    y1: float = -1

    class Config:
        orm_mode = False


class SDUElement(BaseModel):
    id: int
    start: int = -1
    end: int = -1

    class Config:
        orm_mode = False


class SDUAttachment(BaseModel):
    id: str = ""
    name: str = ""
    path: str = ""
    meta: Dict = {}
    text: SDUText = SDUText()
    charset: str = ""
    encoding: str = ""
    disposition: str = ""
    content_type: str = ""
    binary: bool = False
    payload: str = ""
    status: str = ""

    class Config:
        orm_mode = False


class SDUDimensions(BaseModel):
    id: int = -1
    height: float = 0.0
    width: float = 0.0
    factor_x: float = 0.0
    factor_y: float = 0.0
    rotation: int = 0

    class Config:
        orm_mode = False


class SDUFonts(BaseModel):
    id: int = -1
    fontsizes: Dict = {}
    fonts: List = []
    avg_fontsize: int = 14
    small_fontsize: int = 10000

    class Config:
        orm_mode = False


class SDULayout(BaseModel):
    id: int = -1
    dimensions: SDUDimensions = SDUDimensions()
    fonts: SDUFonts = SDUFonts()
    texttrace: List = []
    images: List = []
    drawings: List = []
    blocks: List[tuple] = []
    bjson: Dict = {}
    columns: List[SDUBBox] = []
    rows: List[SDUBBox] = []
    layouts: List = []
    header: SDUBBox = SDUBBox()
    body: SDUBBox = SDUBBox()
    footer: SDUBBox = SDUBBox()
    margin_left: SDUBBox = SDUBBox()
    margin_right: SDUBBox = SDUBBox()

    class Config:
        orm_mode = False


class SDUContent(BaseModel):
    attachments: List[SDUAttachment] = []
    layouts: List[SDULayout] = []

    class Config:
        orm_mode = False


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
    Input document model made over SDULanguage. Default language english

    Attributes:
        language: object SDULanguage.
    """

    language: SDULanguage = SDULanguage(code="en", lang="english")


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


class SPKLanguageDTO(SDULanguageDetails):
    """DTO, representing the result of service language."""


class TextWithParagraphsGet(BaseModel):
    """
    Schema representing the result of paragraph segmentation.

    Attributes:
        paragraphs: list of SDUParagraph.
    """

    paragraphs: List[SDUParagraph]


class TextWithSentencesGet(BaseModel):
    """
    Schema representing the result of sentences segmentation.

    Attributes:
        sentences: list of SDUSentence.
    """

    sentences: List[SDUSentence]


class TextWithPagesGet(BaseModel):
    """
    Schema representing the result of pages segmentation.

    Attributes:
        pages: list of SDUPage.
    """

    pages: List[SDUPage]


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
    language: SDULanguage = SDULanguage(code="en", lang="english")


class SPKSegmentationDTO(BaseModel):
    """
    DTO, representing the result of service segmentation. Only one attribute will be non-empty.

    Attributes:
        pages: list of SDUPage.
        paragraphs: list of SDUParagraph.
        sentences: list of SDUSentence.
    """

    pages: List[SDUPage] = []
    paragraphs: List[SDUParagraph] = []
    sentences: List[SDUSentence] = []


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

        data: List of nested dictionaries
        keys: The keys  which need to clean
        language: default is german
    """

    data: List[Dict[str, Dict[str, Any]]]
    keys: List[str] = []
    language: SDULanguage = SDULanguage(code="de", lang="german")


class SPKDataCleanAIDTO(BaseModel):
    """
    DTO, representing the result of service ai text clean.

    Attributes:

        data: LList of nested dictionaries
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
        language: default is german
    """

    data: List[Dict[str, Dict[str, Any]]]
    algorithms: List[str] = ["yake", "bert"]
    keys: List[str] = []
    language: SDULanguage = SDULanguage(code="de", lang="german")


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


class SPKStatisticsDTO(SDUStatistic):
    """DTO, representing the result of service Statistics."""


class SPKSummaryDTO(wdc.WDCItem):
    """DTO, representing the result of service Summary."""


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
        model_data: train columns, features, etc
    """

    info: str
    id: Optional[uuid.UUID]
    path: Optional[str]
    model_data: Optional[Dict]


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
        webhook_url: url to custom HTTP back requests.
    """

    model_name: str = "kim_pipeline"
    data: List[Dict[str, Any]]
    target_columns: List[str] = ["IMPULSART", "IMPULSKATEGORIE"]
    train_columns: List[str] = []
    text_features: List[str] = ["SACHVERHALT", "SACHVERHALT_KEYWORDS"]
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
    webhook_url: Optional[str]


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
        full_file_path: path to file.
        layout_file_path: path to layout file.
        debug_file_path: path to debug file.
        readorder_file_path: path to rearorder file.
        clean_text_path: path to txt file with clean text.
        raw_text_path: path to txt file with raw text.
        html_path: path to txt file with html.
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
        images: images.
        pages_layout: layouts.
        pages_text: pages.
        content: content.
        metadata: metadata.
        description: discription.
        status: document status
        file: file.
        sdu: Dict of sdu objects.
    """

    uid: str
    name: str
    mimetype: str = "text/plain"
    full_file_path: str = ""
    layout_file_path: str = ""
    debug_file_path: str = ""
    readorder_file_path: str = ""
    clean_text_path: str = ""
    raw_text_path: str = ""
    html_path: str = ""
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
    images: List[SDUPageImage] = []
    pages_layout: List[SDULayout] = []
    pages_text: List[SDUPage] = []
    content: Optional[SDUContent] = None
    metadata: Dict = {}
    description: str = ""
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
