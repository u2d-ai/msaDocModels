import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from bson.objectid import ObjectId
from msaDocModels import wdc
from pydantic import UUID4, BaseModel, Field


def to_camel(string: str) -> str:
    """
    Converts snake_case string to camelCase

    Parameters:

        string: text to convert

    Returns:

        string in camelCase
    """
    split_string = string.split("_")

    return "".join([split_string[0]] + [word.capitalize() for word in split_string[1:]])


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


class ResultType(str, Enum):
    document = "document"
    pages = "pages"
    paragraphs = "paragraphs"
    sentences = "sentences"


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


class ExtractionDefaultResult(BaseModel):
    """
    Model for representing an extracted entity.

    Attributes:
        id: id of founded entity.
        text: The text of the extracted entity.
        start: The start index of the extracted entity in the input text.
        end: The end index of the extracted entity in the input text.
    """

    id: int
    text: str
    start: int
    end: int


class ExtractionExtendedDefaultResult(ExtractionDefaultResult):
    """
    Model for representing an extracted entity with a name to which the result is related.

    Attributes:

        name: The name to which the result refers.
    """

    name: str


class RecognizerDefaultResult(ExtractionDefaultResult):
    """
    Model for representing a recognized entity.

    Attributes:
        type: The type of the recognized entity.
    """

    type: str


class TextExtractionFormats(BaseModel):
    """
    Data transfer object for entity extractor.

    Attributes:

        recognizer: A list of RecognizerDefaultResult objects representing the results obtained from entity recognition.
        result: A dictionary representing results extracted by regex patterns.
            The keys of the dictionary are unique identifiers for the extracted data,
            and the values are lists of ExtractionDefaultResult or ExtractionExtendedDefaultResult objects.
    """

    recognizer: List[RecognizerDefaultResult] = []
    result: Dict[str, List[Union[ExtractionExtendedDefaultResult, ExtractionDefaultResult]]] = {}


class TextExtractionFormatsStrDTO(BaseModel):
    """DTO, representing the result of extraction Formats for Str"""

    extractions: TextExtractionFormats


class TextExtractionFormatsListDTO(BaseModel):
    """DTO, representing the result of extraction Formats for List"""

    extractions: List[TextExtractionFormats]


class TextExtractionFormatsDictDTO(BaseModel):
    """DTO, representing the result of extraction Formats for Dict"""

    extractions: Dict[str, TextExtractionFormats]


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
    """
    Detected Language Pydantic Model.

    Attributes:
        code: Short de, en etc.
        lang: Language name like german.
        reliable: is the detected result reliable.
        proportion:  Proportion of the text in this language.
        bytes:  Bytes of the text in this language.
        confidence: Confidence from 0.01 to 1.0.
        winner: Selected overall Winner
    """

    code: str = "unknown"
    lang: str = "unknown"
    reliable: bool = False
    proportion: int = -1
    bytes: int = -1
    confidence: float = -1
    winner: Optional[str] = None

    class Config:
        orm_mode = False


class SDULanguageDetails(SDULanguage):
    """
    Detected Detail Language Pydantic Model.

    Attributes:
        details: Details of the top 3 detected language
    """

    details: Optional[List] = []


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
    """Sentence Pydantic Model."""

    id: int = -1
    text: str = ""
    lang: SDULanguage = SDULanguage()

    class Config:
        orm_mode = False


class SDUPDFElement(BaseModel):
    """PDF Element Pydantic Model."""

    line_id: int = -1
    span_id: int = -1
    flags: int = 0
    bold: bool = False
    italic: bool = False
    font: str = ""
    fontsize: float = 0.0
    color: int = 0


class SDUParagraph(BaseModel):
    """Paragraph Pydantic Model."""

    id: int = -1
    sort: int = -1
    nsen: int = 0
    semantic_type: str = "text"
    section: str = "body"
    size_type: str = "body"
    sentences: List[SDUSentence] = []
    lang: SDULanguage = SDULanguage()

    class Config:
        orm_mode = False

    def has_text(self) -> bool:
        return len(self.sentences) > 0

    def get_text(self) -> str:
        return self.get_text_no_lf()

    def get_paragraph_text(self) -> str:
        paragraph_text = ""
        for sentence in self.sentences:
            paragraph_text += sentence.text
        return paragraph_text

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
    """Text Pydantic Model."""

    raw: str = ""
    clean: str = ""
    html_content: str = ""
    raw_path: str = ""
    clean_path: str = ""
    html_path: str = ""
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
    """Learnset Pydantic Model."""

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


class NotaryInput(DocumentInput):
    """
    Data input model for Notary.

    Attributes:
        city: default city to check.
    """

    city: str = "Bremen"


class NotaryRemovalValidity(BaseModel):
    """
    Notary removal validity field.

    Attributes:

        from_: start date
        to: end date
    """

    from_: Any
    to: Any


class NotaryInterruptionValidity(BaseModel):
    """
    Notary interruption validity field.

    Attributes:

        from_: start date
        to: end date
    """

    from_: Any
    to: Any


class NotaryValidity(BaseModel):
    """
    Notary validity field.

    Attributes:

        from_: start date
        to: end date
    """

    from_: Any
    to: Any


class NotaryItem(BaseModel):
    """
    Class that represents record in notary.json file.

    Attributes:

        id: Notary identifier.
        sid: Security identifier in format "str(ret.id) + "@" + str(ret.person_id)"
        tmp_office_rm_from: Temporary office rm from.
        tmp_office_rm_to: Temporary office rm to.
        chamber_name: Chamber name
        removal_validity: Notary removal validity.
        tmp_office_rm_status: Temporary office rm status.
        tmp_office_pause_from: Temporary office pause from.
        tmp_office_pause_to: Temporary office pause to.
        interruption_validity: Notary interruption validity.
        tmp_office_pause_status: Temporary office pause status.
        chamber_id: Chamber id.
        last_name: Last name of the notary.
        historical_names: Historical Names.
        first_name: Fist anme of the notary.
        office_title: Notary office title.
        office_title_full: Notary full office title.
        office_title_code: Notary Office title code.
        zip_code: Zip code.
        city: City in which notary os located.
        office_city: City in which office os located.
        official_location: Notary official location.
        address: Notary address.
        additional_address: Notary additional address.
        title: Notary title.
        title_appendix: Notary title appendix.
        phone: Notary phone.
        language1: Fist notary language.
        language2: Secondary notary language.
        complete_name_with_official_location: Complete Notary name with office location.
        url: Notary website
        valid_to: The license is valid to
        valid_from: The license is valid from
        notary_validity: Notary validity
        office_hour_cities: Office hour cities.
        historical_cities: Historical cities.
        latitude: Office latitude.
        longitude: Office langitude.
        user_id: User identifier.
        group_id: Group identifier.
        person_id: Person identifier.
        is_bremen: Is office located in bremen.
    """

    id: int = -1
    sid: Optional[str] = ""
    tmp_office_rm_from: Any
    tmp_office_rm_to: Any
    chamber_name: str = "na"
    removal_validity: NotaryRemovalValidity
    tmp_office_rm_status: Any
    tmp_office_pause_from: Any
    tmp_office_pause_to: Any
    interruption_validity: NotaryInterruptionValidity
    tmp_office_pause_status: Any
    chamber_id: int = -1
    last_name: Optional[str] = ""
    historical_names: Any
    first_name: Optional[str] = ""
    office_title: Optional[str] = ""
    office_title_full: Optional[str] = ""
    office_title_code: Optional[str] = ""
    zip_code: Optional[str] = ""
    city: Optional[str] = ""
    office_city: Optional[str] = ""
    official_location: Optional[str] = ""
    address: Optional[str] = ""
    additional_address: Optional[str] = ""
    title: Any
    title_appendix: Any
    phone: Optional[str] = ""
    language1: Any
    language2: Any
    complete_name_with_official_location: Optional[str] = ""
    url: Any
    valid_to: Any
    valid_from: Optional[str] = ""
    notary_validity: NotaryValidity
    office_hour_cities: Any
    historical_cities: Optional[str] = ""
    latitude: Any
    longitude: Any
    user_id: Optional[str] = ""
    group_id: Optional[str] = ""
    person_id: int = -1
    is_bremen: Optional[bool] = False

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True

    def get_train_data(self, chambers_only: bool = False) -> str:
        """
        The function generates a feature string based on the values of various properties of the object.

        Parameters:

            chambers_only: If True, includes only the zip code, city, address, historical cities, and office city.
            If False, includes the city, address, first name, last name, and office title full. Formats: False.

        Returns:

            string
        """
        xt_feature = ""

        if chambers_only:
            for item in (self.zip_code, self.city, self.address, self.historical_cities, self.office_city):
                if item:
                    xt_feature += " " + item
        else:
            for item in (self.city, self.address, self.first_name, self.last_name, self.office_title_full):
                if item:
                    xt_feature += " " + item

        return xt_feature


class Notary(BaseModel):
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


class NotaryWinnerDTO(Notary):
    """DTO, representing the result of service Notary."""


class SDUPage(BaseModel):
    """Page Pydantic Model."""

    id: int = -1
    npar: int = 0
    input: str = ""
    text: SDUText = SDUText()

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

    def get_page_text(self) -> str:
        page_text = ""
        for paragraph in self.text.paragraphs:
            for sentence in paragraph.sentences:
                page_text += sentence.text
        return page_text

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
                if par.semantic_type.__contains__("head") or par.semantic_type.__contains__("title"):
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

    def set_input(self, input_text: str):
        self.input = input_text


class BaseDocumentInput(BaseModel):
    """
    Data input model for extraction from document.

    Attributes:
        pages_text: The document data.
        document_id: optional uuid for document.
    """

    pages_text: List[SDUPage] = []
    document_id: Optional[UUID4]

    def get_document_text(self) -> str:
        document_text = ""
        for page in self.pages_text:
            for paragraph in page.text.paragraphs:
                for sentence in paragraph.sentences:
                    document_text += sentence.text
        return document_text

    def get_list_document_strings(self) -> List[str]:
        strings = []
        for page in self.pages_text:
            for paragraph in page.text.paragraphs:
                for sentence in paragraph.sentences:
                    strings.append(sentence.text)
        return strings

    def get_list_paragraph_strings(self) -> List[str]:
        strings = []
        for page in self.pages_text:
            for paragraph in page.text.paragraphs:
                strings.append(paragraph.get_paragraph_text())
        return strings

    def get_list_page_strings(self) -> List[str]:
        strings = []
        for page in self.pages_text:
            strings.append(page.get_page_text())
        return strings

    def get_document_lang(self) -> str:
        if self.pages_text:
            return self.pages_text[0].text.lang.code
        return "unknown"


class ExtractKeywordsDocumentInput(BaseDocumentInput):
    """
    Data input model for ExtractorKeywords.

    Attributes:
        result_output: Type of output format.
        algorithms: which algorithms use for extract. Can be list of ["yake", "bert", "bert_vectorized", "tf_idf"]
    """

    result_output: ResultType = ResultType.sentences
    algorithms: List[str] = ["yake", "bert"]


class SDUData(BaseModel):
    """Data Pydantic Model."""

    npages: int = 0  #
    stats: SDUStatistic = SDUStatistic()  #
    pages: List[SDUPage] = []
    converter: List[str] = []
    email: SDUEmail = SDUEmail()
    text: SDUText = SDUText()
    images: List[SDUPageImage] = []

    class Config:
        orm_mode = False

    def add_page_pre_processing(self, page_pre: SDUPage):
        if page_pre:
            self.pages.append(page_pre)
            self.npages = len(self.pages)
            page_pre.page = self.npages


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
    blocks: Dict = {}
    columns: List[SDUBBox] = []
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


class LanguageInput(DocumentInput):
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


class LanguageDTO(SDULanguageDetails):
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


class SegmentationInput(BaseModel):
    """
    Input model to detect segmentation

    Attributes:
        document_id: optional uuid for document.
        input_text: input_text.

    """

    document_id: Optional[UUID4]
    input_text: Union[str, List[str], Dict[int, str]]


class SegmentationDTO(BaseModel):
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


class TextCleanInput(DocumentInput):
    """Data input model for Text Clean."""


class TextCleanDTO(BaseModel):
    """
    DTO, representing the result of service text clean.

    Attributes:
        text: cleaned text.
    """

    text: str


class DataCleanAIInput(BaseModel):
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


class DataCleanAIDTO(BaseModel):
    """
    DTO, representing the result of service ai text clean.

    Attributes:

        data: LList of nested dictionaries
    """

    data: List[Dict[str, Dict[str, Any]]]


class SentimentInput(DocumentInput):
    """Data input model for Sentiment."""


class SentimentDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorSentiment.

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


class PhrasesWordBagInput(DocumentLangInput):
    """Data input model for word bag."""


class PhrasesMiningInput(DocumentInput):
    """
    Frequent pattern mining followed by agglomerative clustering on the input corpus

    Attributes:
        min_support: Minimum support threshold which must be satisfied by each phrase during frequent pattern mining.
        max_phrase_size: Maximum allowed phrase size.
        alpha: Threshold for the significance score.
    """

    min_support: int = 10
    max_phrase_size: int = 40
    alpha: int = 4


class KeywordsExtractionParams(DocumentLangInput):
    """
    Input model to extract keywords

    Attributes:
        keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
        top_n: Return the top n keywords/keyphrases
        use_maxsum: Whether to use Max Sum Distance for the selection of keywords/keyphrases.
        use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the selection of keywords/keyphrases.
        diversity: The diversity of the results between 0 and 1 if `use_mmr` is set to True.
        nr_candidates: The number of candidates to consider if `use_maxsum` is set to True.
    """

    keyphrase_ngram_range: Tuple[int, int] = (1, 1)
    top_n: int = 5
    use_maxsum: bool = False
    use_mmr: bool = False
    diversity: float = 0.5
    nr_candidates: int = 20


class PhrasesKeyTermsInput(KeywordsExtractionParams):
    """Data input model for key terms."""


class PhrasesContribInput(KeywordsExtractionParams):
    """Data input model for phrases contrib."""


class PhrasesRakeInput(DocumentLangInput):
    """Data input model for phrases rake."""


class PhraseKeyword(BaseModel):
    """
    Model that contains keyword data.

    Attributes:
        entity: keyword of document.
        distance: keyword distance to the input document.
    """

    entity: str
    distance: float


class PhrasesKeyTermsDTO(BaseModel):
    """
    DTO, representing the result of key terms.

    Attributes:
        phrases: List of key terms
    """

    phrases: Union[List[PhraseKeyword], List[List[PhraseKeyword]], Dict[Any, List[PhraseKeyword]]]


class PhrasesContribDTO(BaseModel):
    """
    DTO, representing the result of phrases contrib.

    Attributes:
        phrases: List of phrases contribution
    """

    phrases: Union[List[PhraseKeyword], List[List[PhraseKeyword]], Dict[Any, List[PhraseKeyword]]]


class PhrasesRakeDTO(BaseModel):
    """
    DTO, representing the result of phrases rake.

    Attributes:
        phrases: List of most common words
    """

    phrases: Union[List[PhraseKeyword], List[List[PhraseKeyword]], Dict[Any, List[PhraseKeyword]]]


class PhraseMiningDTO(BaseModel):
    """
    DTO, representing the result of phrases mining.

    Attributes:
        partitioned_docs: Document.
        index_vocab: Vocabulary for text.
    """

    partitioned_docs: List[List[List[int]]]
    index_vocab: List[Any]


class PhrasesWordBagDTO(BaseModel):
    """
    DTO, representing the result of word bag.

    Attributes:
        phrases: Nested list of most common phrases in the provided sentence(s)
    """

    phrases: List[Union[List, List[Union[str, int]]]]


class WeightedKeywordsDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorKeywords.

    Attributes:
        keywords:  List of keywords and/or keyphrases.
    """

    keywords: List[Union[List, List[Union[str, int]]]]


class ExtractKeywordsInput(BaseModel):
    """
    Data input model for ExtractorKeywords.

    Attributes:
        data: extended input text by InputKeyKeys, have the len as input.
        algorithms: which algorithms use for extract. Can be list of ["yake", "bert", "bert_vectorized", "tf_idf"]
        keys: which keys need to extract
        language: default is german
    """

    data: List[Dict[str, Dict[str, Any]]]
    algorithms: List[str] = ["yake", "bert"]
    keys: List[str] = []
    language: SDULanguage = SDULanguage(code="de", lang="german")


class ExtractKeywordsTextInput(DocumentInput):
    """
    Data input model for ExtractorKeywords.

    Attributes:
        algorithms: which algorithms use for extract. Can be list of ["yake", "bert", "bert_vectorized", "tf_idf"]
        language: default is german
    """

    algorithms: List[str] = ["yake", "bert"]
    language: SDULanguage = SDULanguage(code="de", lang="german")


class ExtractKeywordsTextDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorKeywords.

    Attributes:
        data: Extracted keywords for text.
    """

    data: Union[List[str], List[List[str]], Dict[Any, List[str]]]


class ExtractKeywordsDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorKeywords.

    Attributes:
        data: extended input text by InputKeyKeys, have the len as input.
    """

    data: List[Dict[str, Dict[str, Any]]]


class SummaryInput(DocumentLangInput):
    """
    Data input model for EngineSummary.

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


class StatisticsInput(DocumentLangInput):
    """Data input model for ExtractorStatistics."""


class StatisticsDTO(SDUStatistic):
    """DTO, representing the result of service ExtractorStatistics."""


class SummaryEmbeddedInput(DocumentLangInput):
    """Data input model for EngineSummary Embedded."""


class SentenceTopicsInput(DocumentLangInput):
    """
    Data input model for Sentence topics.

    Attributes:

            multiplier: Multiplier used for increasing the size of the training data using synthetic samples.
    """

    multiplier: int = 20


class TopicFrequency(BaseModel):
    """Frequency for sentence in topic.

    Attributes:
        sentence: topic sentence.
        frequency: sentence frequency in topic.
    """

    sentence: str
    frequency: float


class TopicInfo(BaseModel):
    """
    Information about topic including it frequency and name.

    Attributes:
        name: topic name.
        sentences: topic sentences and their frequency.
    """

    name: str
    sentences: List[TopicFrequency]


class SentenceTopicsDTO(BaseModel):
    """
    DTO, representing the result of service Sentence Topics.

    Attributes:
        topics: list of information about each topic.
        visuals: topics visuals.
    """

    topics: List[TopicInfo]
    visuals: List[str]


class SentenceSummary(BaseModel):
    """Sentence along with its respective rate."""

    sentence: str
    rate: float


class SummaryEmbeddedDTO(BaseModel):
    """DTO, representing the result of service EngineSummary Embedded.
    Attributes:
        sentences_summary: List of sentences along with their respective rates.
    """

    sentences_summary: List[SentenceSummary]


class SummaryDTO(wdc.WDCItem):
    """DTO, representing the result of service EngineSummary."""


class Country(BaseModel):
    """
    Detected Country Pydantic Model.

    Attributes:
        name: name
        official: official
        currencies: currencies
        capital:capital
        region: region
        subregion: subregion
        languages: languages
        latlng: latlng
        flag: flag
        calling_codes: calling_codes
    """

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


class Company(BaseModel):
    """
    Detected Company Pydantic Model.

    Attributes:
        rank: rank
        company: name
        employees: employees
        change_in_rank: change_in_rank
        industry: industry
        description: description
        revenue: revenue
        revenue_change: revenue_change
        profits: profits
        profit_change: profit_change
        assets: assets
        market_value: market_value

    """

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


class City(BaseModel):
    """Detected City Pydantic Model."""

    name: str
    country: str
    latlng: List[float]


class TaxonomyCitiesDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorTaxonomy cities.

    Attributes:

        cities: List of Cities.
        cities_winner: winner object City.
    """

    cities: List[City]
    cities_winner: Optional[City]


class TaxonomyCountriesDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorTaxonomy countries.

    Attributes:

        countries: List of Countries.
        countries_winner: winner object Country.
    """

    countries: List[Country]
    countries_winner: Optional[Country]


class TaxonomyCompaniesDTO(BaseModel):
    """
    DTO, representing the result of service ExtractorTaxonomy companies.

    Attributes:

        companies: List of Companies.
        companies_winner: winner object Company.
    """

    companies: List[Company]
    companies_winner: Optional[Company]


class TaxonomyDTO(TaxonomyCountriesDTO, TaxonomyCompaniesDTO, TaxonomyCitiesDTO):
    """DTO, representing the result of service ExtractorTaxonomy."""


class TaxonomyInput(DocumentInput):
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


class ProfileInput(BaseModel):
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


class ProfileDTO(BaseModel):
    """
    Pydantic model of Profile HTML representation
    """


class BuildModelInput(BaseModel):
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


class InferenceInput(BaseModel):
    """
     Pydantic model for get inference data.

    Attributes:

        path: The path where model is located.
        data: Profile html representation.
    """

    path: str
    data: List[Dict[str, Any]]


class InferenceDTO(BaseModel):
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

    number: str = "000.000.000.000"
    timestamp: datetime = str(datetime.utcnow())


class DBBaseDocumentInput(BaseModel):
    """
    Document fields for input.

    Attributes:

        uid: document uid
        name: document name.
        mimetype: mimetype.
        full_file_path: path to file.
        debug_file_path: path to debug file.
        readorder_file_path: path to readorder file.
        clean_text_path: path to txt file with clean text.
        raw_text_path: path to txt file with raw text.
        html_path: path to txt file with html.
        output_file_paths: paths to output files.
        folder: folder name.
        group_uuid: group identifier.
        tags: list of tags.
        language: language.
        needs_update: need update or not.
        data: data.
        images: images.
        pages_layout: layouts.
        pages_text: pages.
        metadata: metadata.
        description: description.
        status: document status
        file: file.
        sdu: Dict of sdu objects.
    """

    uid: str
    name: str
    mimetype: str = "text/plain"
    email_file_path: str = ""
    full_file_path: str = ""
    debug_file_path: str = ""
    readorder_file_path: str = ""
    clean_text_path: str = ""
    raw_text_path: str = ""
    html_path: str = ""
    output_file_paths: List[str] = []
    folder: str = ""
    group_uuid: str = ""
    project: str = ""
    tags: Optional[List] = []
    language: Optional[SDULanguage] = None
    needs_update: bool = False
    data: Dict = {}
    images: List[SDUPageImage] = []
    pages_layout: List[SDULayout] = []
    pages_text: List[SDUPage] = []
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


class DBBaseDocumentDTO(DBBaseDocumentInput, MongoId):
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
        key: object name.
    """

    version: str
    description: str
    datetime: datetime
    key: str


class UpdateAI(BaseModel):
    """
    Update ai fields.

    Attributes:
        version: version identifier.
        description: description.
        datetime: datetime.
        key: object name.
    """

    version: Optional[str]
    description: Optional[str]
    datetime: Optional[datetime]
    key: Optional[str]


class LearnsetDataInput(BaseInfo):
    """
    AI learnset input.

    Attributes:

        learnsets: list of learnset objects or learnset object.
    """

    learnsets: Union[List[Dict], Dict]


class TestsetDataInput(BaseInfo):
    """
    AI testset input.

    Attributes:

        testsets: list of testsets objects or learnset object.
    """

    testsets: Union[List[Dict], Dict]


class ModelDataInput(BaseInfo):
    """
    AI model input.

    Attributes:

        model: model object.
    """

    model: Dict


class TestsetDataDTO(TestsetDataInput, MongoId):
    """
    AI testset output.
    """


class LearnsetDataDTO(LearnsetDataInput, MongoId):
    """
    AI learnset output.
    """


class ModelDataDTO(ModelDataInput, MongoId):
    """
    AI model output.
    """


class TestsetsDataDTO(MongoId):
    """
    AI testsets output.

    Attributes:
        testsets: list of testsets object.
    """

    testsets: List[TestsetDataInput]


class LearnsetsDataDTO(MongoId):
    """
    AI learns output.

    Attributes:
        learnsets: list of learnsets object.
    """

    learnsets: List[LearnsetDataInput]


class ModelsDataDTO(MongoId):
    """
    AI models output.
    Attributes:
        models: list of models object.
    """

    models: List[ModelDataInput]


class ConvertToXLSXInput(BaseModel):
    """
    Model that contains inference data along with filenames to use for XLSX conversion.

    Attributes:

        file_paths: list of file paths (can include filenames or filenames with directories) to be saved
        inference: inference data, first key means sheet name for XLSX file
    """

    file_paths: List[str]
    inference: List[Dict[str, Dict[str, Any]]]


class ConvertToZIPInput(BaseModel):
    """
    Model that contains filespaths which need to save in zip.

    Attributes:

        file_paths: list of list of file paths to be saved.
        zip_names: list of names of archives
    """

    file_paths: List[List[str]]
    zip_names: List[str]


class HTMLConverterResponse(BaseModel):
    """
    Response from converter

    Attributes:

        metadata: metadata from file
        txt_content: SDUText object
    """

    metadata: Dict
    txt_content: SDUText


class EmailConverterResponse(BaseModel):
    """
    Matching pydantic models with fields in the db.

    Attributes:

        content_attachments: list of SDUAttachments.
        embedding_attachments: list of inline SDUAttachments.
        txt_content: SDUText.
        msg: SDUEmail.
        content_unzipped_files: object of unzipped files.
        email_tags: segmented email by tags
    """

    content_attachments: List[SDUAttachment]
    embedding_attachments: List[SDUAttachment]
    txt_content: SDUText
    msg: SDUEmail
    content_unzipped_files: Optional[List[HTMLConverterResponse]]
    email_tags: Dict = {}


class EmailConverterWithoutAttachmentsResponse(EmailConverterResponse):
    content_attachments: Optional[List[SDUAttachment]]
    embedding_attachments: Optional[List[SDUAttachment]]


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


class SaveConfigInputModel(BaseModel):
    """
    Input data model for "save-config" router.

    Attributes:

        input_data: json config.
    """

    input_data: Dict


class SaveAIDataInputModel(BaseModel):
    """
    Input data model for "save-ai-data" router.

    Attributes:

        subdomain: db name(tenant).
        ai_data: testset / learnset.
    """

    ai_data: Union[TestsetDataInput, LearnsetDataInput]
    subdomain: str


class SaveAIModelInputModel(BaseModel):
    """
    Input data model for "save-ai-model" router.

    Attributes:

        ai_data: model.
        subdomain: db name(tenant).
    """

    ai_data: ModelDataInput
    subdomain: str


class UpdateAIModelInputModel(BaseModel):
    """
    Input data model for "update-testset"/"update-model"/"update-learnset" routers.

    Attributes:

        subdomain: db name(tenant).
        version: testset version.
        key: name of testset.
        data_for_update: field for update.
    """

    subdomain: str
    version: str
    key: str
    data_for_update: UpdateAI


class SaveFullDocumentInputModel(BaseModel):
    """
    Input data model for "save-full-document" router.

    Attributes:

        input_data: document fields.
        subdomain: db name(tenant).
        client_id: collection name(user identifier).
    """

    input_data: DBBaseDocumentInput
    subdomain: str
    client_id: str


class UpdateSDUFieldInputModel(BaseModel):
    """
    Input data model for "update-sdu-field" / "update-sdu-list-field" routers.

    Attributes:

        input_data: Dict that contains data to be processed.
        subdomain: db name(tenant).
        client_id: collection name(user identifier).
        document_uid: uid of the document to be updated.
    """

    input_data: Dict
    subdomain: str
    client_id: str
    document_uid: str


class UpdateStatusInputModel(BaseModel):
    """
    Input data model for "update-status" router.

    Attributes:

        process_status: number of the process status.
        subdomain: db name(tenant).
        client_id: collection name(user identifier).
        document_uid: uid of the document.
    """

    process_status: str
    subdomain: str
    client_id: str
    document_uid: str


class UpdateDocumentInputModel(BaseModel):
    """
    Input data model for "update-document-fields" router.

    Attributes:

        subdomain: db name (tenant)
        client_id: collection name(user identifier).
        document_uid: uid of the document to be updated.
        fields_for_update: fields with new data
    """

    subdomain: str
    client_id: str
    document_uid: str
    fields_for_update: Dict


class FilterByStatusInputModel(BaseModel):
    """
    Input data model for "filter-by-status" router.
    Attributes:
        subdomain: db name (tenant)
        client_id: collection name (user identifier)
        document_uid: uid of the document
        project: name of the project
        status_lower_bound: status of document should be greater than this parameter (X in the x <= status <= y)
        status_upper_bound: status of documents should be lower than this parameter (Y in the x <= status <= y)
        one_document: only one document if True
        update_status: change status document, when return
    """

    subdomain: Optional[str] = None
    client_id: Optional[str] = None
    document_uid: Optional[str] = None
    project: Optional[str] = None
    status_lower_bound: Optional[str] = "000.000.000.000"
    status_upper_bound: Optional[str] = "999.999.999.999"
    update_status: Optional[str] = None
    one_document: bool = False


class CheckStatusHistoryInputModel(BaseModel):
    """
    Input data model for "check-status-history" router.
    Attributes:
        status: status of document
        status_lower_bound: status of document should be greater than this parameter (X in the x <= status <= y)
        status_upper_bound: status of documents should be lower than this parameter (Y in the x <= status <= y)
        subdomain: db name (tenant)
        client_id: collection name (user identifier)
        document_uid: uid of the document
        project: name of the project
        update_status: change status document, when return
    """

    status: Optional[str] = None
    status_lower_bound: Optional[str] = None
    status_upper_bound: Optional[str] = None
    subdomain: Optional[str] = None
    client_id: Optional[str] = None
    document_uid: Optional[str] = None
    project: Optional[str] = None
    update_status: Optional[str] = None


class EntityExtractorInput(DocumentLangInput):
    """
    Model that contains input data for extract Formats.

    Attributes:

        regex to extract data.
    """

    patterns: Optional[Dict]


class EntityExtractorDocumentInput(BaseDocumentInput):
    """
    Model that contains input data for extract Formats.

    Attributes:

        patterns: regex to extract data.
        result_output: Type of output format.
    """

    patterns: Optional[Dict]
    result_output: ResultType = ResultType.sentences


class TextExtractionNLPInput(DocumentInput):
    """
    Data input model for extraction NLP from text.
    """


class TextExtractionDocumentNLPInput(BaseDocumentInput):
    """
    Data input model for extraction NLP from document.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.sentences


class ExtractionNLP(BaseModel):
    """
    Model which represent find NLP in text.

    Attributes:

        id: id of founded entity.
        text: found entity
        upos: upos in text.
        xpos: xpos in text.
        feats: features of entity.
        s: start char in text.
        e: end char in text.
    """

    id: int
    text: str
    upos: str
    xpos: str
    feats: Optional[str]
    s: int
    e: int


class TextExtractionNLPDTO(BaseModel):
    """
    Data input model for ExtractionNLP.

    Attributes:

        extractions: List of ExtractionNLP.
    """

    extractions: Union[List[ExtractionNLP], List[List[ExtractionNLP]], Dict[Any, List[ExtractionNLP]]]


class TextExtractionDocumentFormatsDTO(BaseModel):
    """
    Model that contains extraction data implemented in page data.

    Attributes:

        pages_text: The document data with extractions with the same structure.
    """

    pages_text: List[SDUPage] = []


class NestingId(BaseModel):
    """
    Model that represents id of nesting structure (page, paragraph, sentence..)

    Attributes:

        id: id of found entity.
    """

    id: int = -1


class SentenceNLPDTO(NestingId):
    """
    Model that represents a sentences with NLP extractions.

    Attributes:

        result: list of sentences with nlp found in the page.
    """

    result: List[ExtractionNLP] = []


class ParagraphNLPDTO(NestingId):
    """
    Model that represents a paragraph with NLP extractions.

    Attributes:

        sentences: list of sentences.
    """

    sentences: List[SentenceNLPDTO] = []


class PageNLPDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        paragraphs: list of paragraphs.
    """

    paragraphs: List[ParagraphNLPDTO] = []


class TextExtractionDocumentNLPPage(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with NLP extractions.
    """

    version: str
    pages_text: List[PageNLPDTO] = []


class Position(BaseModel):
    """
    Model that represents a position.

    Attributes:
        s: The start index of a position.
        e: The end index of a position.
    """

    s: int
    e: int


class ExtractionNER(NestingId):
    """
    Model that represents named entity found in text.

    Attributes:

        text: found entity
        type: recognition of text.
    """

    text: str
    type: str
    positions: List[Position]


class PageNERDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        result: list of named entity recognition extractions found in the page.
    """

    result: List[ExtractionNER] = []


class TextExtractionNERInput(BaseModel):
    """
    Model that represents an input for named entity recognition text extraction.

    Attributes:

        input_text: input text.
        language: object SDULanguage.
    """

    input_text: Union[str, List[str], Dict[Any, str]]
    language: SDULanguage = SDULanguage(code="de", lang="geman")


class TextExtractionDocumentNERInput(BaseDocumentInput):
    """
    Model that represents an input for named entity recognition text extraction on a document.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.sentences


class TextExtractionNERDTO(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction.

    Attributes:

        result: list of named entity recognition extractions found in the text. Can be a list of extractions,
                    a list of lists of extractions (for multiple documents), or a dictionary with keys
                    representing document ids and values representing lists of extractions.
    """

    result: Union[List[ExtractionNER], List[List[ExtractionNER]], Dict[Any, List[ExtractionNER]]]


class TextExtractionDocumentNERDTO(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with named entity recognition extractions.
    """

    version: str
    pages_text: List[PageNERDTO] = []


class SentenceFormatsDTO(NestingId):
    """
    Model that represents a sentences with ExtractionFormats extractions.

    Attributes:

        result: list of sentences with ExtractionFormats found in the page.
    """

    result: TextExtractionFormats


class ParagraphFormatsDTO(NestingId):
    """
    Model that represents a paragraph with ExtractionFormats extractions.

    Attributes:

        sentences: list of sentences.
    """

    sentences: List[SentenceFormatsDTO] = []


class PageFormatsDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        paragraphs: list of paragraphs.
    """

    paragraphs: List[ParagraphFormatsDTO] = []


class TextExtractionDocumentFormatsPage(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with ExtractionFormats extractions.
    """

    version: str
    pages_text: List[PageFormatsDTO] = []


class PageNotaryDTO(NestingId):
    """
    Model that represents a page with notary extractions.

    Attributes:

        result: Notary object if found notary.
    """

    result: Union[Notary, Dict]


class TextExtractionNotaryDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: Notary object if found notary or empty dict.
    """

    version: str
    result: Union[Notary, Dict]


class InformationExtractionAnswerTextInput(DocumentInput):
    """
    Model that represents an input for extraction information from answers from text.

    Attributes:
        questions: questions about context.
        limit: max number of answers.
    """

    questions: List[str] = []
    limit: int = 1


class InformationExtractionAnswerDocumentInput(BaseDocumentInput):
    """
    Model that represents an input for extraction information from answers from document.

    Attributes:
        result_output: Type of output format.
        questions: questions about context.
        limit: max number of answers.
    """

    result_output: ResultType = ResultType.pages
    questions: List[str] = []
    limit: int = 1


class AnswerExtraction(BaseModel):
    """
    Model that represents data of extraction information from answer

    Attributes:
        answer: answer about context.
        score: number of predict.
        s: answer start position.
        e: answer end position.
    """

    answer: str
    score: float
    s: int
    e: int


class InformationExtractionAnswerTextDTO(BaseModel):
    """
    Model that contains data extraction information from answers.

    Attributes:
        result: extracted answers.
    """

    result: Union[List[AnswerExtraction], List[List[AnswerExtraction]], Dict[Any, List[AnswerExtraction]]]


class InformationExtractionQuestionTextInput(DocumentInput):
    """
    Model that represents an input for extraction information from questions from text.

    Attributes:
        answers: answers about context.
        max_length: max length of question.
    """

    answers: List[str] = []
    max_length: int = 64


class InformationExtractionQuestionDocumentInput(BaseDocumentInput):
    """
    Model that represents an input for extraction information from questions from document.

    Attributes:
        result_output: Type of output format.
        answers: answers about context.
        max_length: max length of question.
    """

    result_output: ResultType = ResultType.pages
    answers: List[str] = []
    max_length: int = 64


class InformationExtractionQuestionTextDTO(BaseModel):
    """
    Model that contains data extraction information from questions.

    Attributes:
        result: extracted questions.
    """

    result: Union[List[str], List[List[str]], Dict[Any, List[str]]]


class SentenceAnswerInformationDTO(NestingId):
    """
    Model that represents a sentences with answer information extractions.

    Attributes:
        result: list of sentences with answers found in the sentence of page.
    """

    result: List[AnswerExtraction] = []


class ParagraphAnswerInformationDTO(NestingId):
    """
    Model that represents a paragraph with answer information extractions.
    Attributes:
        sentences: list of sentences.
    """

    sentences: List[SentenceAnswerInformationDTO] = []


class ParagraphAnswerInformationResult(NestingId):
    """
    Model that represents a sentences with answer information extractions.

    Attributes:
        result: list of sentences with answers found in Paragraph.
    """

    result: List[AnswerExtraction]


class PageAnswerInformationDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:
        paragraphs: list of paragraphs.
    """

    paragraphs: Union[List[ParagraphAnswerInformationResult], List[ParagraphAnswerInformationDTO]]


class PageAnswerInformationResult(NestingId):
    """
    Model that represents a sentences with answer information extractions.

    Attributes:
        result: list of sentences with answers found in page.
    """

    result: List[AnswerExtraction]


class InformationExtractionAnswerDocumentPage(BaseModel):
    """
    Model that represents the result of search answer in context.

    Attributes:
        version: version of the text extraction service used.
        pages_text: list of pages with answer information extractions.
    """

    version: str
    pages_text: Union[List[PageAnswerInformationResult], List[PageAnswerInformationDTO]]


class InformationExtractionAnswerDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text answer on a document for all doc.

    Attributes:

        version: version of the answer extraction service used.
        result: list of sentences with answers found in the doc.
    """

    version: str
    result: List[AnswerExtraction]


class InformationExtractionAnswerDocumentDTO(BaseModel):
    """
    Model that contains answers data implemented in page data.

    Attributes:
        information_extraction_answer: The same structure with document.
    """

    information_extraction_answer: Union[InformationExtractionAnswerDocument, InformationExtractionAnswerDocumentPage]


class InformationExtractionAnswerPageDocumentPage(BaseModel):
    """
    Model that represents the result of search answer in page.

    Attributes:
        version: version of the text extraction service used.
        pages_text: list of pages with answer information extractions.
    """

    version: str
    pages_text: List[SentenceAnswerInformationDTO] = []


class InformationExtractionAnswerPageDocumentDTO(BaseModel):
    """
    Model that contains answers data implemented in page data.

    Attributes:
        information_extraction_answer: The same structure with document.
    """

    information_extraction_answer: InformationExtractionAnswerPageDocumentPage


class SentenceQuestionInformationDTO(NestingId):
    """
    Model that represents a sentences with question information extractions.

    Attributes:
        result: list of sentences with questions found in the sentence of page.
    """

    result: List[str]


class ParagraphQuestionInformationDTO(NestingId):
    """
    Model that represents a paragraph with question information extractions.

    Attributes:
        sentences: list of sentences.
    """

    sentences: List[SentenceQuestionInformationDTO]


class ParagraphQuestionInformationResult(NestingId):
    """
    Model that represents a sentences with question information extractions.

    Attributes:
        result: list of sentences with question found in Paragraph.
    """

    result: List[str]


class PageQuestionInformationDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.
    Attributes:
        paragraphs: list of paragraphs.
    """

    paragraphs: Union[List[ParagraphQuestionInformationResult], List[ParagraphQuestionInformationDTO]]


class PageQuestionInformationResult(NestingId):
    """
    Model that represents a sentences with question information extractions.

    Attributes:
        result: list of sentences with question found in page.
    """

    result: List[str]


class InformationExtractionQuestionDocumentPage(BaseModel):
    """
    Model that represents the result of search question in context.
    Attributes:
        version: version of the text extraction service used.
        pages_text: list of pages with question information extractions.
    """

    version: str
    pages_text: Union[List[PageQuestionInformationResult], List[PageQuestionInformationDTO]]


class InformationExtractionQuestionDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text question on a document for all doc.

    Attributes:

        version: version of the Question extraction service used.
        result: list of sentences with Question found in the doc.
    """

    version: str
    result: List[str]


class InformationExtractionQuestionDocumentDTO(BaseModel):
    """
    Model that contains questions data implemented in page data.
    Attributes:
        information_extraction_question: The same structure with document.
    """

    information_extraction_question: Union[
        InformationExtractionQuestionDocument, InformationExtractionQuestionDocumentPage
    ]


class InformationExtractionQuestionPageDocumentPage(BaseModel):
    """
    Model that represents the result of search question in page.
    Attributes:
        version: version of the text extraction service used.
        pages_text: list of pages with question information extractions.
    """

    version: str
    pages_text: List[SentenceQuestionInformationDTO] = []


class InformationExtractionQuestionPageDocumentDTO(BaseModel):
    """
    Model that contains questions data implemented in page data.
    Attributes:
        information_extraction_question: The same structure with document.
    """

    information_extraction_question: InformationExtractionQuestionPageDocumentPage


class InformationExtractionAnswerInput(BaseModel):
    """
    Data input model for answer.
    Attributes:
        question: question about context
        context: text which using as context for question
    """

    question: str
    context: str


class InformationExtractionQuestionInput(BaseModel):
    """
    Data input model for question.
    Attributes:
        answer: answer about context
        context: text which using as context for answer
        max_length: max length of answer
    """

    answer: str
    context: str
    max_length: int = 64


class KeywordsDocument(BaseModel):
    """
    Model that represents the result of Keywords analysis on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list of Keywords.
    """

    version: str
    result: List


class KeywordsPageResult(NestingId):
    """
    Model that represents the result of Keywords analysis on a document for page.

    Attributes:

        result: list of Keywords.
    """

    result: List


class KeywordsParagraphResult(NestingId):
    """
    Model that represents the result of Keywords analysis on a document for paragraph.

    Attributes:

        result: list of Keywords.
    """

    result: List


class KeywordsSentenceResult(NestingId):
    """
    Model that represents the result of keywords analysis on a document for sentence.

    Attributes:

        result: list of Keywords.
    """

    result: List


class KeywordsParagraphSentences(NestingId):
    """
    Model that represents the list of sentences with result of keywords analysis on a document.

    Attributes:

        sentences: list of sentences with result of keywords analysis extractions.
    """

    sentences: List[KeywordsSentenceResult]


class KeywordsPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs with result of keywords analysis on a document.

    Attributes:

        paragraphs: list of paragraphs with result of keywords analysis extractions.
    """

    paragraphs: Union[List[KeywordsParagraphResult], List[KeywordsParagraphSentences]]


class KeywordsPages(BaseModel):
    """
    Model that represents the list of pages with result of keywords analysis on a document.

    Attributes:

        version: version of the keywords analysis service used.
        pages_text: list of pages with result of keywords analysis extractions.
    """

    version: str
    pages_text: Union[List[KeywordsPageResult], List[KeywordsPageParagraphs]]


class KeywordsDocumentDTO(BaseModel):
    """
    Model that contains result of keywords analysis implemented in sentence/paragraph/page/doc data.

    Attributes:

            text_keywords: The same structure with document.

    """

    text_keywords: Union[KeywordsDocument, KeywordsPages]


class ConciseData(BaseModel):
    """
    Model that represents concise data.

    Attributes:

        concept_data: A dictionary of concept data.
        upper_data: A dictionary of upper data.
        input_data: A dictionary of input data.
        patterns: A list of patterns.
        concise_config: A dictionary of Concise configuration data.
    """

    concept_data: Dict[str, List[str]]
    upper_data: Dict[str, List[str]]
    input_data: Dict[str, List[str]]
    patterns: List
    concise_config: Dict


class ConciseElement(BaseModel):
    """
    Model that represents a concise element.

    Attributes:

        path: the path where model is located.
        name: name of the Concise element.
        lang: language of the Concise element.
        data: ConciseData instance containing the data for the Concise element.
    """

    path: str
    name: str
    lang: str
    data: ConciseData


class EntityInfo(BaseModel):
    """
    Model that represent founded entities.

    Attributes:

        hits: The number of occurrences .
        avg: The Average number of predict.
        min: The Minimal number of predict.
        max: The Maximum number of predict.
        high: The High number of predict.

    """

    hits: int
    avg: float
    min: float
    max: float
    high: float


class TextDomainsInput(BaseModel):
    """
    Model that represents an input for named entity recognition text extraction.

    Attributes:

        input_text: input text.
    """

    input_text: Union[str, List[str], Dict[Any, str]]


class DomainEntity(BaseModel):
    """
    Model that represent founded entities.

    Attributes:

        entity: The entity.
        scores: The object of scores.

    """

    entity: str
    scores: EntityInfo


class ClassifierEntity(BaseModel):
    """
    Model that represent founded Entities.
    Attributes:
        start: start ent
        end: end ent
        label: The label.
        scores: The object of scores.
    """

    start: int
    end: int
    label: str
    scores: EntityInfo
    tooltip: str
    kb_id: str


class DocClassifierEntity(BaseModel):
    """
    Model that represent founded Entities.

    Attributes:

        entity: The entity.
        scores: The object of scores.

    """

    entity: str
    scores: EntityInfo


class PageConciseDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        result: PreparedConciseResult object
    """

    result: List[ClassifierEntity]


class ParagraphResultConciseDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: list of named entity recognition extractions found in the paragraph.
    """

    result: List[ClassifierEntity]


class SentenceResultConciseDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: list of named entity recognition extractions found in the sentence.
    """

    result: List[ClassifierEntity]


class ParagraphSentencesConciseDTO(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with concise concept extractions.
    """

    sentences: List[SentenceResultConciseDTO]


class PageParagraphsConciseDTO(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with concise concept extractions.
    """

    paragraphs: Union[List[ParagraphResultConciseDTO], List[ParagraphSentencesConciseDTO]]


class TextExtractionConciseDocumentPage(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with concise concept result.
    """

    version: str
    model_version: str
    pages_text: Union[List[PageConciseDTO], List[PageParagraphsConciseDTO]]


class TextExtractionConciseDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction concise on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list of ClassifierEntity objects.
    """

    version: str
    model_version: str
    result: List[ClassifierEntity]


class TextExtractionConciseDocumentDTO(BaseModel):
    """
    DTO, representing the result of concise concepts

    Attributes:

        text_extraction_concise: TextExtractionConciseDocumentPage object
    """

    text_extraction_concise: Union[TextExtractionConciseDocumentPage, TextExtractionConciseDocument]


class TextExtractionDocumentConciseConceptsInput(BaseDocumentInput):
    """
    Data input model for extraction EngineConciseConcept from document.

    Attributes:
        result_output: Type of output format.
        path: The path where model is located.

    """

    result_output: ResultType = ResultType.pages
    path: str = ""


class TextExtractionTextConciseConceptsInput(BaseModel):
    """
    Data input model for extraction EngineConciseConcept from text.

    Attributes:

        path: The path where model is located.
                input_text: The document data.

    """

    path: str = ""
    input_text: str


class TrainDocumentConciseConceptsInput(BaseModel):
    """
    Data input model for train EngineConciseConcept from document.

    Attributes:

        data: A dictionary containing the training data. The keys are concept names and the values are
              lists of strings.
        use_wordnet_enrichment: A boolean value indicating whether to use WordNet enrichment. Default is False.
        element_name: The name of the element to train. Default is "LoeBi".
        lang: The language of the input data. Default is "de".
        version: The version of the model to train. Default is "v1".
        top_default: The number of words to be returned for each class.
        verbose: Use verbose formatting.
        exclude_pos: A list of POS tags to exclude from the rule based match
        exclude_dep: list of dependencies to exclude from the rule based match
        include_compound_words: If True, it will include compound words in the entity. For example,
                                    if the entity is "New York", it will also include "New York City" as an entity,
                                    Formats to False (optional)
        case_sensitive: Whether to match the case of the words in the text, Formats to False (optional)
        fuzzy: If True, it will be use fuzzy matching formatting.
        entities_threshold: Threshold to include entities.
        verbose: Use verbose formatting.
    """

    data: Dict
    element_name: str = "LoeBi"
    lang: str = "de"
    version: str = "v1"
    use_wordnet_enrichment: bool = False
    top_default: int = 100
    verbose: bool = False
    exclude_pos: List[str] = ["VERB", "AUX"]
    exclude_dep: List[str] = ["DOBJ", "PCOMP"]
    include_compound_words: bool = False
    case_sensitive: bool = False
    fuzzy: bool = True
    entities_threshold: float = 0.3


class SentimentDocumentInput(BaseDocumentInput):
    """
    Model that contains input data to sentiment analysis.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.sentences


class SentimentDocument(BaseModel):
    """
    Model that represents the result of sentiment analysis on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: SentimentDTO object.
    """

    version: str
    result: SentimentDTO


class SentimentPageResult(NestingId):
    """
    Model that represents the result of sentiment analysis on a document for page.

    Attributes:

        result: SentimentDTO object.
    """

    result: SentimentDTO


class SentimentParagraphResult(NestingId):
    """
    Model that represents the result of sentiment analysis on a document for paragraph.

    Attributes:

        result: SentimentDTO object.
    """

    result: SentimentDTO


class SentimentSentenceResult(NestingId):
    """
    Model that represents the result of sentiment analysis on a document for sentence.

    Attributes:

        result: SentimentDTO object.
    """

    result: SentimentDTO


class SentimentParagraphSentences(NestingId):
    """
    Model that represents the list of sentences with result of sentiment analysis on a document.

    Attributes:

        sentences: list of sentences with result of sentiment analysis extractions.
    """

    sentences: List[SentimentSentenceResult]


class SentimentPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs with result of sentiment analysis on a document.

    Attributes:

        paragraphs: list of paragraphs with result of sentiment analysis extractions.
    """

    paragraphs: Union[List[SentimentParagraphResult], List[SentimentParagraphSentences]]


class SentimentPages(BaseModel):
    """
    Model that represents the list of pages with result of sentiment analysis on a document.

    Attributes:

        version: version of the sentiment analysis service used.
        pages_text: list of pages with result of sentiment analysis extractions.
    """

    version: str
    pages_text: Union[List[SentimentPageResult], List[SentimentPageParagraphs]]


class SentimentDocumentDTO(BaseModel):
    """
    Model that contains result of sentiment analysis implemented in sentence/paragraph/page/doc data.

    Attributes:

            text_sentiment: The same structure with document.

    """

    text_sentiment: Union[SentimentDocument, SentimentPages]


class TextLanguageDocumentInput(BaseDocumentInput):
    """
    Model that contains input data to detect language.

    Attributes:
        result_output: Type of output format.
        hint_languages: language hint for analyzer. 'ITALIAN' or 'it' boosts Italian;see LANGUAGES for known languages.
        hint_encoding: encoding hint for analyzer. 'SJS' boosts Japanese; see ENCODINGS for all known encodings.
        sentence_detection: turn on/off language detection by sentence.
        get_vectors: turn on/off return of sentence vectors.
        is_plain_text: if turned off, and HTML is provided as input, detection will skip HTML tags,
            expand HTML entities and detect HTML <lang ...> tags.
        is_short_text: turn on to get the best effort results (instead of unknown) for short text.

    """

    result_output: ResultType = ResultType.pages
    hint_languages: str = ""
    hint_encoding: str = ""
    sentence_detection: bool = True
    get_vectors: bool = True
    is_plain_text: bool = True
    is_short_text: bool = False


class TextLanguageDocument(BaseModel):
    """
    Model that represents the result of detected language on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: LanguageDTO object.
    """

    version: str
    result: LanguageDTO


class TextLanguagePageResult(NestingId):
    """
    Model that represents the result of detected language on a document for page.

    Attributes:

        result: LanguageDTO object.
    """

    result: LanguageDTO


class TextLanguageParagraphResult(NestingId):
    """
    Model that represents the result of detected language on a document for paragraph.

    Attributes:

        result: LanguageDTO object.
    """

    result: LanguageDTO


class TextLanguageSentenceResult(NestingId):
    """
    Model that represents the result of detected language on a document for sentence.

    Attributes:

        result: LanguageDTO object.
    """

    result: LanguageDTO


class TextLanguageParagraphSentences(NestingId):
    """
    Model that represents the list of sentences with result of detected language on a document.

    Attributes:

        sentences: list of sentences with result of detected language extractions.
    """

    sentences: List[TextLanguageSentenceResult]


class TextLanguagePageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs with result of detected language on a document.

    Attributes:

        paragraphs: list of paragraphs with result of detected language extractions.
    """

    paragraphs: Union[List[TextLanguageParagraphResult], List[TextLanguageParagraphSentences]]


class TextLanguagePages(BaseModel):
    """
    Model that represents the list of pages with result of detected language on a document.

    Attributes:

        version: version of the text language service used.
        pages_text: list of pages with result of detected language extractions.
    """

    version: str
    pages_text: Union[List[TextLanguagePageResult], List[TextLanguagePageParagraphs]]


class TextLanguageDocumentDTO(BaseModel):
    """
    Model that contains result of detected language implemented in sentence/paragraph/page/doc data.

    Attributes:

            text_language: The same structure with document.

    """

    text_language: Union[TextLanguageDocument, TextLanguagePages]


class StatisticsDocument(BaseModel):
    """
    Model that represents the result of named entity recognition ExtractorStatistics on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: SDUStatistic object.
    """

    version: str
    result: SDUStatistic


class StatisticsPageResult(NestingId):
    """
    Model that represents the result of named entity recognition ExtractorStatistics on a document for page.

    Attributes:

        result: SDUStatistic object.
    """

    result: SDUStatistic


class StatisticsParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition ExtractorStatistics on a document for paragraph.

    Attributes:

        result: SDUStatistic object.
    """

    result: SDUStatistic


class StatisticsSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition ExtractorStatistics on a document for sentence.

    Attributes:

        result: SDUStatistic object.
    """

    result: SDUStatistic


class StatisticsParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of ExtractorStatistics on a document.

    Attributes:

        sentences: list of sentences with ExtractorStatistics extractions.
    """

    sentences: List[StatisticsSentenceResult]


class StatisticsPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of ExtractorStatistics on a document.

    Attributes:

        paragraphs: list of paragraphs with ExtractorStatistics extractions.
    """

    paragraphs: Union[List[StatisticsParagraphResult], List[StatisticsParagraphSentences]]


class StatisticsPages(BaseModel):
    """
    Model that represents the list of pages of ExtractorStatistics on a document.

    Attributes:

        version: version of the ExtractorStatistics service used.
        pages_text: list of pages with ExtractorStatistics extractions.
    """

    version: str
    pages_text: Union[List[StatisticsPageResult], List[StatisticsPageParagraphs]]


class StatisticsDocumentDTO(BaseModel):
    """
    Model that contains ExtractorStatistics data implemented in sentence/paragraph/page/doc data.

    Attributes:

            text_extraction_statistics: The same structure with document.

    """

    text_extraction_statistics: Union[StatisticsDocument, StatisticsPages]


class CleanDocumentInput(BaseDocumentInput):
    """
    Model that contains input data to clean text from document.

    Attributes:
        language: language of text
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.pages
    language: SDULanguage = SDULanguage(code="de", lang="german")


class PageCleanDTO(NestingId):
    """
    Model that represents a page with clean document.

    Attributes:

        result: clean string.
    """

    result: str


class CleanParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineClean on a document for paragraph.

    Attributes:

        result: clean string.
    """

    result: str


class CleanSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineClean on a document for sentence.

    Attributes:

        result: clean string.
    """

    result: str


class CleanParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of EngineClean on a document.

    Attributes:

        sentences: list of sentences with EngineClean extractions.
    """

    sentences: List[CleanSentenceResult]


class CleanPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of EngineClean on a document.

    Attributes:

        paragraphs: list of paragraphs with EngineClean extractions.
    """

    paragraphs: Union[List[CleanParagraphResult], List[CleanParagraphSentences]]


class PageClean(BaseModel):
    """
    Model that represents the result of EngineClean of a document.

    Attributes:

        version: version of the translation service used.
        pages_text: list of pages with document EngineClean.
    """

    version: str
    pages_text: Union[List[PageCleanDTO], List[CleanPageParagraphs]]


class CleanDocument(BaseModel):
    """
    Model that represents the result of named entity recognition EngineClean on a document for all doc.

    Attributes:

        version: version of the EngineClean service used.
        result: EngineClean string.
    """

    version: str
    result: str


class CleanDocumentDTO(BaseModel):
    """
    Model that contains Document EngineClean data per page.

    Attributes:

        text_clean: The same structure with document.

    """

    text_clean: Union[CleanDocument, PageClean]


class TranslateTextInput(BaseModel):
    """
    Model that contains input data to translate text.

    Attributes:
        input_text: str, contains text to translate
        from_lang: str, language from which text will be translated (language of text)
        to_lang: str, language to which text will be translated
        split_underscore: bool, if text is split by underscore

    """

    input_text: Union[str, List[str], Dict[Any, str]]
    from_lang: Optional[str] = None
    to_lang: Optional[str] = None
    split_underscore: bool = False


class TranslateDocumentInput(BaseDocumentInput):
    """
    Model that contains input data to translate text from document.

    Attributes:
        result_output: Type of output format.
        from_lang: language from which text will be translated (language of text)
        to_lang: language to which text will be translated
        split_underscore: split underscore or no
    """

    result_output: ResultType = ResultType.pages
    from_lang: Optional[str] = None
    to_lang: Optional[str] = None
    split_underscore: bool = False


class TranslateEntity(BaseModel):
    """
    Model that contains input data to translate text from document.

    Attributes:
        input_text: str, text to translate
        translated_text: str, translated text
        from_lang: str, language from which text will be translated
        to_lang: str, language to which text will be translated
        split_underscore: bool, does text contain underscores instead of spaces
    """

    input_text: str
    translated_text: str
    from_lang: str
    to_lang: str
    split_underscore: bool


class TranslateTextDTO(BaseModel):
    """
    Model that represents the result of translation.

    Attributes:
        result: result of translation
    """

    result: Union[TranslateEntity, List[TranslateEntity], Dict[Any, TranslateEntity]]


class PageTranslationDTO(NestingId):
    """
    Model that represents a page with translated document.

    Attributes:

        result: TranslateEntityDTO.
    """

    result: TranslateEntity


class TranslationParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text translate on a document for paragraph.

    Attributes:

        result: TranslateEntity object.
    """

    result: TranslateEntity


class TranslationSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition text translation on a document for sentence.

    Attributes:

        result: TranslateEntity object.
    """

    result: TranslateEntity


class TranslationParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of text translation on a document.

    Attributes:

        sentences: list of sentences with EngineTranslation extractions.
    """

    sentences: List[TranslationSentenceResult]


class TranslationPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of text translation on a document.

    Attributes:

        paragraphs: list of paragraphs with EngineTranslation extractions.
    """

    paragraphs: Union[List[TranslationParagraphResult], List[TranslationParagraphSentences]]


class PageTranslation(BaseModel):
    """
    Model that represents the result of translation of a document.

    Attributes:

        version: version of the translation service used.
        pages_text: list of pages with document EngineTranslation extractions.
    """

    version: str
    pages_text: Union[List[PageTranslationDTO], List[TranslationPageParagraphs]]


class TranslationDocument(BaseModel):
    """
    Model that represents the result of named entity recognition EngineTranslation on a document for all doc.

    Attributes:

        version: version of the EngineTranslation service used.
        result: TranslateEntity object.
    """

    version: str
    result: TranslateEntity


class TranslateDocumentDTO(BaseModel):
    """
    Model that contains Document EngineTranslation data.

    Attributes:

        text_translation: The same structure with document.

    """

    text_translation: Union[TranslationDocument, PageTranslation]


class AISearchInputModel(BaseModel):
    """
    Input data model for "ai-search" router.

    Attributes:

        document_id: ID of the document.
        input_text: The input text to search. Can be a string, list, or dictionary.
    """

    document_id: Optional[str] = None
    input_text: Union[str, List, Dict]


class AISearchOutputModel(BaseModel):
    """
    Output data model for "ai-search" router.

    Attributes:

        document_id: ID of the document.
        data: Notary object.
    """

    document_id: Optional[str] = None
    data: Notary


class ExtractionDocumentNotaryInput(BaseDocumentInput):
    """
    Data input model for extraction Notary from document.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.pages


class ParagraphResultNotaryDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: Notary object if found notary or empty dict.
    """

    result: Union[Notary, Dict]


class SentenceResultNotaryDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: Notary object if found notary or empty dict.
    """

    result: Union[Notary, Dict]


class ParagraphSentencesNotaryDTO(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with Notary extractions.
    """

    sentences: List[SentenceResultNotaryDTO]


class PageParagraphsNotaryDTO(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with Notary extractions.
    """

    paragraphs: Union[List[ParagraphResultNotaryDTO], List[ParagraphSentencesNotaryDTO]]


class TextExtractionNotaryDocumentPage(BaseModel):
    """
    Model that represents the result of search notary in text.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with Notary extractions.
    """

    version: str
    pages_text: Union[List[PageNotaryDTO], List[PageParagraphsNotaryDTO]]


class TextExtractionNotaryDocumentDTO(BaseModel):
    """
    Model that contains notary data.

    Attributes:

            text_extraction_notary: The same structure with document.

    """

    text_extraction_notary: Union[TextExtractionNotaryDocumentPage, TextExtractionNotaryDocument]


class PredictionFieldNames(str, Enum):
    """
    Enum Class that represents choices for prediction fields.
    """

    hits = "hits"
    avg = "avg"
    min = "min"
    max = "max"
    high = "high"


class DeviceTypes(str, Enum):
    """
    Enum Class that represents choices for device type.
    """

    auto = "auto"
    cuda_0 = "cuda:0"
    cpu = "cpu"


class DocClassifierTextInput(BaseModel):
    """
    Model that represents the input for classification of text.

    Attributes:

        document_id: str
        input_text: text to classify.
        label_structure_data: topics for text classification.
        learnset_name: name of learnset.
        learnset_version: version of learnset.
        learnset_lang: lang of learnset.
        multi_label: does 'label_structure_data' contain more than one topic.
        score_threshold: threshold to include entities.
        use_text_filters: use filters to clean text.
        context_min_length: min length of context.
    """

    document_id: Optional[str]
    input_text: Union[str, List[str], Dict[Any, str]]
    label_structure_data: Dict[str, List[str]]
    learnset_name: str = ""
    learnset_version: str = ""
    learnset_lang: str = ""
    multi_label: bool = True
    score_threshold: float = 0.3
    use_text_filters: bool = True
    context_min_length: int = 50


class DocClassifierDocumentInput(BaseDocumentInput):
    """
    Model that represents the input for classification of document.

    Attributes:

        document_id: str
        result_output: Type of output format.
        label_structure_data: topics for text classification.
        learnset_name: name of learnset.
        learnset_version: version of learnset.
        learnset_lang: lang of learnset.
        multi_label: does 'label_structure_data' contain more than one topic.
        score_threshold: threshold to include entities.
        use_text_filters: use filters to clean text.
        context_min_length: min length of context.
    """

    document_id: Optional[str]
    result_output: ResultType = ResultType.pages
    label_structure_data: Dict[str, List[str]]
    learnset_name: str = ""
    learnset_version: str = ""
    learnset_lang: str = ""
    multi_label: bool = True
    score_threshold: float = 0.3
    use_text_filters: bool = True
    context_min_length: int = 50


class DocClassifierDTO(BaseModel):
    """
    Model that represents result of classification of the model.

    Attributes:

        prediction: result of classification of the model.
    """

    prediction: Union[List[DocClassifierEntity], List[List[DocClassifierEntity]], Dict[Any, List[DocClassifierEntity]]]


class PageClassifierDTO(NestingId):
    """
    Model that represents a page with document prediction.

    Attributes:

        result: list ClassifierEntity.
    """

    result: List[DocClassifierEntity]


class ClassifierParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: list of ClassifierEntity objects.
    """

    result: List[DocClassifierEntity]


class ClassifierSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: list of ClassifierEntity objects.
    """

    result: List[DocClassifierEntity]


class ClassifierParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with Classifier extractions.
    """

    sentences: List[ClassifierSentenceResult]


class ClassifierPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with Classifier extractions.
    """

    paragraphs: Union[List[ClassifierParagraphResult], List[ClassifierParagraphSentences]]


class DocClassifierPage(BaseModel):
    """
    Model that represents the result classifier on a document.

    Attributes:

        version: version of the classifier service used.
        learnset_version: version of the learnset used.
        pages_text: list of pages with document prediction.
    """

    version: str
    learnset_version: str
    pages_text: Union[List[PageClassifierDTO], List[ClassifierPageParagraphs]]


class DocClassifierDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list of ClassifierEntity objects.
    """

    version: str
    result: List[DocClassifierEntity]


class DocClassifierDocumentDTO(BaseModel):
    """
    Model that contains document prediction data.

    Attributes:

        doc_classifier: The same structure with document.

    """

    doc_classifier: Union[DocClassifierDocument, DocClassifierPage]


class DomainsExtractorDocumentInput(BaseDocumentInput):
    """
    Model that contains input data for extract formats.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.sentences


class TextDomainsDTO(BaseModel):
    """
    Data input model for EngineClean.

    Attributes:

        extractions: List of ExtractionDomains.
    """

    extractions: Union[List[DomainEntity], List[List[DomainEntity]], Dict[Any, List[DomainEntity]]]


class SentenceDomainsDTO(NestingId):
    """
    Model that represents a sentences with domains extractions.

    Attributes:

        result: list of sentences with domains found in the page.
    """

    result: List[DomainEntity]


class TextExtractionDomainsParagraphSentences(NestingId):
    """
    Model that represents a paragraph with domains extractions.

    Attributes:

        sentences: list of sentences.
    """

    sentences: List[SentenceDomainsDTO]


class TextExtractionDomainsParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: list of DomainEntity objects.
    """

    result: List[DomainEntity]


class TextExtractionDomainsPageParagraphs(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        paragraphs: list of paragraphs.
    """

    paragraphs: Union[List[TextExtractionDomainsParagraphResult], List[TextExtractionDomainsParagraphSentences]]


class TextExtractionDomainsPageResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for page.

    Attributes:

        result: list of DomainEntity objects.
    """

    result: List[DomainEntity]


class TextExtractionDomainsPages(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with Domains extractions.
    """

    version: str
    pages_text: Union[List[TextExtractionDomainsPageResult], List[TextExtractionDomainsPageParagraphs]]


class TextExtractionDomainsDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list of DomainEntity objects.
    """

    version: str
    result: List[DomainEntity]


class TextDomainsDocumentDTO(BaseModel):
    """
    Model that contains Domains data implemented in sentence data.

    Attributes:

        text_extraction_domains: The same structure with document.

    """

    text_extraction_domains: Union[TextExtractionDomainsDocument, TextExtractionDomainsPages]


class DomainsData(BaseModel):
    """
    Model that contains domains data.

    Attributes:

        domains_flat: English wordnet domains.
        domains_flat_de: German wordnet domains.
        synsets_by_domain: English Synsets for wordnet.
        lemmas_by_domain: English Lemmas for wordnet.
        synsets_by_domain_de: German Synsets for wordnet.
        lemmas_by_domain_de: German Lemmas for wordnet.

    """

    domains_flat: List[str] = []
    domains_flat_de: List[str] = []
    synsets_by_domain: Dict[str, List[str]] = {}
    lemmas_by_domain: Dict[str, List[str]] = {}
    synsets_by_domain_de: Dict[str, List[str]] = {}
    lemmas_by_domain_de: Dict[str, List[str]] = {}


class SearchResultNodeBase(BaseModel):
    """
    BaseModel for search result nodes.

    Attributes:

        name: entry name.
        tooltip: hover tooltip for his node
    """

    name: str
    tooltip: str
    itemStyle: dict = {}


class SearchResultLeaf(SearchResultNodeBase):
    """
    Leaf Model for search result nodes with no parents.

    Attributes:

        value: number, needs to be 1 for echart.
    """

    value: int = 1


class SearchResultNode(SearchResultNodeBase):
    """
    Node Model for search result nodes with parents.

    Attributes:

        name: entry name.
        tooltip: hover tooltip for his node
    """

    children: List


class LemmasData(BaseModel):
    """
    Model that get/represents a lemmas.

    Attributes:

        data: dict of lemmas with list of words for each lemma.
    """

    data: Dict
    lang: str = "de"


class SearchInputData(BaseModel):
    """
    Model that get/represents a search input.

    Attributes:

        word: str word to search.
        lang: str language to use for search (Wordnet languages)
    """

    word: str
    lang: str = "de"


class SearchResultData(BaseModel):
    """
    Model that get/represents a search result.

    Attributes:

        word: word that was searched.
        lang: language that was used for search (Wordnet languages)
        tree: result tree data list of nodes and leafs (children)
        synsets: synsets and there infos
        info: summary/info of search result
        err_msg: empty if search was ok, else error message
    """

    word: str
    lang: str
    tree: List
    synsets: Dict
    info: Dict
    err_msg: str


class LanguageData(BaseModel):
    """
    Model that get/represents a list of languages.

    Attributes:

        data: dict of lemmas with list of words for each lemma.
    """

    data: List


class PageResultTextKnowledgeDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        result: list of named entity recognition extractions found in the page.
    """

    result: Dict


class ParagraphResultTextKnowledgeDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: list of named entity recognition extractions found in the paragraph.
    """

    result: Dict


class SentenceResultTextKnowledgeDTO(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: list of named entity recognition extractions found in the sentence.
    """

    result: Dict


class ParagraphSentencesTextKnowledgeDTO(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with Knowledge extractions.
    """

    sentences: List[SentenceResultTextKnowledgeDTO]


class PageParagraphsTextKnowledgeDTO(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with Knowledge extractions.
    """

    paragraphs: Union[List[ParagraphResultTextKnowledgeDTO], List[ParagraphSentencesTextKnowledgeDTO]]


class TextExtractionKnowledgeDocumentPage(BaseModel):
    """
    Model that represents the result of text knowledge.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with text knowledge extractions.
    """

    version: str
    pages_text: Union[List[PageResultTextKnowledgeDTO], List[PageParagraphsTextKnowledgeDTO]]


class TextKnowledgeDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: TextExtractionFormats object.
    """

    version: str
    result: Dict


class TextKnowledgeDocumentDTO(BaseModel):
    """
    Model that contains text knowledge data implemented in page data.

    Attributes:

        text_extraction_knowledge: The same structure with document.

    """

    text_extraction_knowledge: Union[TextExtractionKnowledgeDocumentPage, TextKnowledgeDocument]


class TextExtractionDocumentKnowledgeInput(BaseDocumentInput):
    """
    Data input model for extraction Knowledge from document.

    Attributes:
        result_output: Type of output format.
    """

    result_output: ResultType = ResultType.pages


class SynsetMeaning(BaseModel):
    """
    Represents the meaning of a synset in WordNet.

    Attributes:

        id: Synset ID.
        s: Synset start offset in the corresponding text.
        e: Synset end offset in the corresponding text.
        token: The concatenated string of all the senses of the synset.
        name: Synset name.
        pos: Part of speech.
        ili: Inter-Lingual Index.
        lexid: Lexical ID.
        lexicalized: True if the synset is lexicalized.
        lexfile: Lexical file.
        definition: Synset definition.
        metadata: Metadata associated with the synset.
        examples: Examples of usage.
        lemmas: Synonym lemmas.
        senses: Sense keys.
        senses_lemmas: Sense keys mapped to their corresponding lemmas.
        words: Words associated with the synset.
        words_lemmas: Words associated with the synset mapped to their corresponding lemmas.
        hypernyms: Hypernyms of the synset.
        hyponyms: Hyponyms of the synset.
        holonyms: Holonyms of the synset.
        meronyms: Meronyms of the synset.
        hypernyms_lemmas: Hypernyms of the synset mapped to their corresponding lemmas.
        hyponyms_lemmas: Hyponyms of the synset mapped to their corresponding lemmas.
        holonyms_lemmas: Holonyms of the synset mapped to their corresponding lemmas.
        meronyms_lemmas: Meronyms of the synset mapped to their corresponding lemmas.
        hypernyms_path: A list of paths to the hypernyms.
        hypernyms_path_lemmas: A list of paths to the hypernyms mapped to their corresponding lemmas.
        root: The topmost synset
        depth_min: The minimum depth of the synset in the hierarchy.
        depth_max: The maximum depth of the synset in the hierarchy.
    """

    id: int = -1
    s: int = -1
    e: int = -1
    token: str = ""
    name: str = ""
    pos: str = ""
    ili: Dict = {}
    lexid: int = -1
    lexicalized: bool = False
    lexfile: str = ""
    definition: str = ""
    metadata: Dict[str, Any] = {}
    examples: List[str] = []
    lemmas: List[str] = []
    senses: List[str] = []
    senses_lemmas: List[str] = []
    words: List[str] = []
    words_lemmas: List[str] = []
    hypernyms: List[str] = []
    hyponyms: List[str] = []
    holonyms: List[str] = []
    meronyms: List[str] = []
    hypernyms_lemmas: List[str] = []
    hyponyms_lemmas: List[str] = []
    holonyms_lemmas: List[str] = []
    meronyms_lemmas: List[str] = []
    hypernyms_path: List[List[str]] = []
    hypernyms_path_lemmas: List[List[str]] = []
    root: str = ""
    depth_min: int = -1
    depth_max: int = -1


class KnowledgeTextInputModel(BaseModel):
    """
    Input model for 'knowledge-text' router

    Attributes:

        input_text: str, dict or list of str
        lang: text language. Default: de
    """

    input_text: Union[str, Dict, List]
    lang: str = "de"


class TextExtractionNLPDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: ExtractionNLP object.
    """

    version: str
    result: List[ExtractionNLP]


class TextExtractionNLPPageResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for page.

    Attributes:

        result: ExtractionNLP object.
    """

    result: List[ExtractionNLP]


class TextExtractionNLPParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: ExtractionNLP object.
    """

    result: List[ExtractionNLP]


class TextExtractionNLPSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: ExtractionNLP object.
    """

    result: List[ExtractionNLP]


class TextExtractionNLPParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with NLP extractions.
    """

    sentences: List[TextExtractionNLPSentenceResult]


class TextExtractionNLPPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with NLP extractions.
    """

    paragraphs: Union[List[TextExtractionNLPParagraphResult], List[TextExtractionNLPParagraphSentences]]


class TextExtractionNLPPages(BaseModel):
    """
    Model that represents the list of pages of text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with NLP extractions.
    """

    version: str
    pages_text: Union[List[TextExtractionNLPPageResult], List[TextExtractionNLPPageParagraphs]]


class TextExtractionNERDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list of ExtractionNER objects.
    """

    version: str
    result: List[ExtractionNER]


class TextExtractionNERPageResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for page.

    Attributes:

        result: list of ExtractionNER objects.
    """

    result: List[ExtractionNER]


class TextExtractionNERParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: list of ExtractionNER objects.
    """

    result: List[ExtractionNER]


class TextExtractionNERSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: list of ExtractionNER objects.
    """

    result: List[ExtractionNER]


class TextExtractionNERParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with NER extractions.
    """

    sentences: List[TextExtractionNERSentenceResult]


class TextExtractionNERPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with NER extractions.
    """

    paragraphs: Union[List[TextExtractionNERParagraphResult], List[TextExtractionNERParagraphSentences]]


class TextExtractionNERPages(BaseModel):
    """
    Model that represents the list of pages of text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with NER extractions.
    """

    version: str
    pages_text: Union[List[TextExtractionNERPageResult], List[TextExtractionNERPageParagraphs]]


class TextExtractionNERDocumentDTO(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        text_extraction_ner: result of named entity recognition text extraction on a document
                            using the TextExtractionService.
    """

    text_extraction_ner: Union[TextExtractionNERDocument, TextExtractionNERPages]


class TextExtractionFormatsDocument(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: TextExtractionFormats object.
    """

    version: str
    result: TextExtractionFormats


class TextExtractionFormatsPageResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for page.

    Attributes:

        result: TextExtractionFormats object.
    """

    result: TextExtractionFormats


class TextExtractionFormatsParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for paragraph.

    Attributes:

        result: TextExtractionFormats object.
    """

    result: TextExtractionFormats


class TextExtractionFormatsSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition text extraction on a document for sentence.

    Attributes:

        result: TextExtractionFormats object.
    """

    result: TextExtractionFormats


class TextExtractionFormatsParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of text extraction on a document.

    Attributes:

        sentences: list of sentences with Formats extractions.
    """

    sentences: List[TextExtractionFormatsSentenceResult]


class TextExtractionFormatsPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of text extraction on a document.

    Attributes:

        paragraphs: list of paragraphs with Formats extractions.
    """

    paragraphs: Union[List[TextExtractionFormatsParagraphResult], List[TextExtractionFormatsParagraphSentences]]


class TextExtractionFormatsPages(BaseModel):
    """
    Model that represents the list of pages of text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with Formats extractions.
    """

    version: str
    pages_text: Union[List[TextExtractionFormatsPageResult], List[TextExtractionFormatsPageParagraphs]]


class TextExtractionFormatsDocumentDTO(BaseModel):
    """
    Model that contains ExtractionFormats data implemented in sentence data.

    Attributes:

            text_extraction_formats: The same structure with document.

    """

    text_extraction_formats: Union[TextExtractionFormatsDocument, TextExtractionFormatsPages]


class TextExtractionNLPDocumentDTO(BaseModel):
    """
    Model that contains nlp data implemented in sentence data.

    Attributes:

        text_extraction_nlp: The same structure with document.

    """

    text_extraction_nlp: Union[TextExtractionNLPDocument, TextExtractionNLPPages]


class SummaryEmbeddedDocumentInput(BaseDocumentInput):
    """Data input model for EngineSummary Embedded.

    Attributes:

        language: object SDULanguage.
        result_output: Type of output format.
    """

    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class EngineSummaryEmbeddedDocument(BaseModel):
    """
    Model that represents the result of named entity recognition EngineSummaryEmbedded on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: SummaryEmbeddedDTO object.
    """

    version: str
    result: SummaryEmbeddedDTO


class EngineSummaryEmbeddedPageResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryEmbedded on a document for page.

    Attributes:

        result: SummaryEmbeddedDTO object.
    """

    result: SummaryEmbeddedDTO


class EngineSummaryEmbeddedParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryEmbedded on a document for paragraph.

    Attributes:

        result: SummaryEmbeddedDTO object.
    """

    result: SummaryEmbeddedDTO


class EngineSummaryEmbeddedSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryEmbedded on a document for sentence.

    Attributes:

        result: SummaryEmbeddedDTO object.
    """

    result: SummaryEmbeddedDTO


class EngineSummaryEmbeddedParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of EngineSummaryEmbedded on a document.

    Attributes:

        sentences: list of sentences with EngineSummaryEmbedded extractions.
    """

    sentences: List[EngineSummaryEmbeddedSentenceResult]


class EngineSummaryEmbeddedPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of EngineSummaryEmbedded on a document.

    Attributes:

        paragraphs: list of paragraphs with EngineSummaryEmbedded extractions.
    """

    paragraphs: Union[List[EngineSummaryEmbeddedParagraphResult], List[EngineSummaryEmbeddedParagraphSentences]]


class EngineSummaryEmbeddedPages(BaseModel):
    """
    Model that represents the list of pages of EngineSummaryEmbedded on a document.

    Attributes:

        version: version of the EngineSummaryEmbedded service used.
        pages_text: list of pages with EngineSummaryEmbedded extractions.
    """

    version: str
    pages_text: Union[List[EngineSummaryEmbeddedPageResult], List[EngineSummaryEmbeddedPageParagraphs]]


class EngineSummaryEmbeddedDocumentDTO(BaseModel):
    """
    Model that contains EngineSummaryEmbedded data implemented in sentence data.

    Attributes:

            engine_summary_embedded: The same structure with document.

    """

    engine_summary_embedded: Union[EngineSummaryEmbeddedDocument, EngineSummaryEmbeddedPages]


class SummaryDocumentInput(BaseDocumentInput):
    """
    Data input model for EngineSummary.

    Attributes:
        sum_ratio: Coefficient.
        sentences_count: Amount of sentences.
        lsa: Algorithm
        corpus_size: Coefficient
        community_size: Coefficient
        cluster_threshold: Coefficient
        language: object SDULanguage.
        result_output: Type of output format.
    """

    sum_ratio: float = 0.2
    sentences_count: int = 15
    lsa: bool = False
    corpus_size: int = 5000
    community_size: int = 5
    cluster_threshold: float = 0.65
    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class EngineSummaryDocument(BaseModel):
    """
    Model that represents the result of named entity recognition EngineSummary on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: list SummaryDTO objects.
    """

    version: str
    result: List[SummaryDTO]


class EngineSummaryPageResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummary on a document for page.

    Attributes:

        result: list SummaryDTO objects.
    """

    result: List[SummaryDTO]


class EngineSummaryParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummary on a document for paragraph.

    Attributes:

        result: list SummaryDTO objects.
    """

    result: List[SummaryDTO]


class EngineSummarySentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummary on a document for sentence.

    Attributes:

        result: list SummaryDTO objects.
    """

    result: List[SummaryDTO]


class EngineSummaryParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of EngineSummary on a document.

    Attributes:

        sentences: list of sentences with EngineSummary extractions.
    """

    sentences: List[EngineSummarySentenceResult]


class EngineSummaryPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of EngineSummary on a document.

    Attributes:

        paragraphs: list of paragraphs with EngineSummary extractions.
    """

    paragraphs: Union[List[EngineSummaryParagraphResult], List[EngineSummaryParagraphSentences]]


class EngineSummaryPages(BaseModel):
    """
    Model that represents the list of pages of EngineSummary on a document.

    Attributes:

        version: version of the EngineSummary service used.
        pages_text: list of pages with EngineSummary extractions.
    """

    version: str
    pages_text: Union[List[EngineSummaryPageResult], List[EngineSummaryPageParagraphs]]


class EngineSummaryDocumentDTO(BaseModel):
    """
    Model that contains EngineSummary data implemented in sentence data.

    Attributes:

            engine_summary: The same structure with document.

    """

    engine_summary: Union[EngineSummaryDocument, EngineSummaryPages]


class SummaryTopicsDocumentInput(BaseDocumentInput):
    """
    Data input model for Doc topics.

    Attributes:

        multiplier: Multiplier used for increasing the size of the training data using synthetic samples.
        language: object SDULanguage.
        result_output: Type of output format.
    """

    multiplier: int = 20
    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class EngineSummaryTopicsDocument(BaseModel):
    """
    Model that represents the result of named entity recognition EngineSummaryTopics on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: SentenceTopicsDTO object.
    """

    version: str
    result: SentenceTopicsDTO


class EngineSummaryTopicsPageResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryTopics on a document for page.

    Attributes:

        result: SentenceTopicsDTO object.
    """

    result: SentenceTopicsDTO


class EngineSummaryTopicsParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryTopics on a document for paragraph.

    Attributes:

        result: SentenceTopicsDTO object.
    """

    result: SentenceTopicsDTO


class EngineSummaryTopicsSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition EngineSummaryTopics on a document for sentence.

    Attributes:

        result: SentenceTopicsDTO object.
    """

    result: SentenceTopicsDTO


class EngineSummaryTopicsParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of EngineSummaryTopics on a document.

    Attributes:

        sentences: list of sentences with EngineSummaryTopics extractions.
    """

    sentences: List[EngineSummaryTopicsSentenceResult]


class EngineSummaryTopicsPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of EngineSummaryTopics on a document.

    Attributes:

        paragraphs: list of paragraphs with EngineSummaryTopics extractions.
    """

    paragraphs: Union[List[EngineSummaryTopicsParagraphResult], List[EngineSummaryTopicsParagraphSentences]]


class EngineSummaryTopicsPages(BaseModel):
    """
    Model that represents the list of pages of EngineSummaryTopics on a document.

    Attributes:

        version: version of the EngineSummaryTopics service used.
        pages_text: list of pages with EngineSummaryTopics extractions.
    """

    version: str
    pages_text: Union[List[EngineSummaryTopicsPageResult], List[EngineSummaryTopicsPageParagraphs]]


class EngineSummaryTopicsDocumentDTO(BaseModel):
    """
    Model that contains EngineSummaryTopics data implemented in sentence data.

    Attributes:

            engine_summary_topics: The same structure with document.

    """

    engine_summary_topics: Union[EngineSummaryTopicsDocument, EngineSummaryTopicsPages]


class PhrasesContribDocumentInput(BaseDocumentInput):
    """
    Input model to extract PhrasesContrib

    Attributes:

        keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
        top_n: Return the top n keywords/keyphrases
        use_maxsum: Whether to use Max Sum Distance for the selection of keywords/keyphrases.
        use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the selection of keywords/keyphrases.
        diversity: The diversity of the results between 0 and 1 if `use_mmr` is set to True.
        nr_candidates: The number of candidates to consider if `use_maxsum` is set to True.
        language: object SDULanguage.
        result_output: Type of output format.
    """

    keyphrase_ngram_range: Tuple[int, int] = (1, 1)
    top_n: int = 5
    use_maxsum: bool = False
    use_mmr: bool = False
    diversity: float = 0.5
    nr_candidates: int = 20
    multiplier: int = 20
    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class PhrasesContribDocument(BaseModel):
    """
    Model that represents the result of named entity recognition PhrasesContrib on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: PhrasesContribDTO object.
    """

    version: str
    result: PhrasesContribDTO


class PhrasesContribPageResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesContrib on a document for page.

    Attributes:

        result: PhrasesContribDTO object.
    """

    result: PhrasesContribDTO


class PhrasesContribParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesContrib on a document for paragraph.

    Attributes:

        result: PhrasesContribDTO object.
    """

    result: PhrasesContribDTO


class PhrasesContribSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesContrib on a document for sentence.

    Attributes:

        result: PhrasesContribDTO object.
    """

    result: PhrasesContribDTO


class PhrasesContribParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of PhrasesContrib on a document.

    Attributes:

        sentences: list of sentences with PhrasesContrib extractions.
    """

    sentences: List[PhrasesContribSentenceResult]


class PhrasesContribPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of PhrasesContrib on a document.

    Attributes:

        paragraphs: list of paragraphs with PhrasesContrib extractions.
    """

    paragraphs: Union[List[PhrasesContribParagraphResult], List[PhrasesContribParagraphSentences]]


class PhrasesContribPages(BaseModel):
    """
    Model that represents the list of pages of PhrasesContrib on a document.

    Attributes:

        version: version of the PhrasesContrib service used.
        pages_text: list of pages with PhrasesContrib extractions.
    """

    version: str
    pages_text: Union[List[PhrasesContribPageResult], List[PhrasesContribPageParagraphs]]


class PhrasesContribDocumentDTO(BaseModel):
    """
    Model that contains PhrasesContrib data implemented in sentence data.

    Attributes:

            extractor_phrases_contrib: The same structure with document.

    """

    extractor_phrases_contrib: Union[PhrasesContribDocument, PhrasesContribPages]


class PhrasesRakeDocumentInput(BaseDocumentInput):
    """
    Input model to extract PhrasesRake

    Attributes:

        language: object SDULanguage.
        result_output: Type of output format.
    """

    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class PhrasesRakeDocument(BaseModel):
    """
    Model that represents the result of named entity recognition PhrasesRake on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: PhrasesRakeDTO object.
    """

    version: str
    result: PhrasesRakeDTO


class PhrasesRakePageResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesRake on a document for page.

    Attributes:

        result: PhrasesRakeDTO object.
    """

    result: PhrasesRakeDTO


class PhrasesRakeParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesRake on a document for paragraph.

    Attributes:

        result: PhrasesRakeDTO object.
    """

    result: PhrasesRakeDTO


class PhrasesRakeSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesRake on a document for sentence.

    Attributes:

        result: PhrasesRakeDTO object.
    """

    result: PhrasesRakeDTO


class PhrasesRakeParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of PhrasesRake on a document.

    Attributes:

        sentences: list of sentences with PhrasesRake extractions.
    """

    sentences: List[PhrasesRakeSentenceResult]


class PhrasesRakePageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of PhrasesRake on a document.

    Attributes:

        paragraphs: list of paragraphs with PhrasesRake extractions.
    """

    paragraphs: Union[List[PhrasesRakeParagraphResult], List[PhrasesRakeParagraphSentences]]


class PhrasesRakePages(BaseModel):
    """
    Model that represents the list of pages of PhrasesRake on a document.

    Attributes:

        version: version of the PhrasesRake service used.
        pages_text: list of pages with PhrasesRake extractions.
    """

    version: str
    pages_text: Union[List[PhrasesRakePageResult], List[PhrasesRakePageParagraphs]]


class PhrasesRakeDocumentDTO(BaseModel):
    """
    Model that contains PhrasesRake data implemented in sentence data.

    Attributes:

            extractor_phrases_rake: The same structure with document.

    """

    extractor_phrases_rake: Union[PhrasesRakeDocument, PhrasesRakePages]


class PhrasesTermsDocumentInput(BaseDocumentInput):
    """
    Input model to extract PhrasesTerms

    Attributes:

        language: object SDULanguage.
        result_output: Type of output format.
    """

    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class PhrasesTermsDocument(BaseModel):
    """
    Model that represents the result of named entity recognition PhrasesTerms on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: PhrasesKeyTermsDTO object.
    """

    version: str
    result: PhrasesKeyTermsDTO


class PhrasesTermsPageResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesTerms on a document for page.

    Attributes:

        result: PhrasesKeyTermsDTO object.
    """

    result: PhrasesKeyTermsDTO


class PhrasesTermsParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesTerms on a document for paragraph.

    Attributes:

        result: PhrasesKeyTermsDTO object.
    """

    result: PhrasesKeyTermsDTO


class PhrasesTermsSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesTerms on a document for sentence.

    Attributes:

        result: PhrasesKeyTermsDTO object.
    """

    result: PhrasesKeyTermsDTO


class PhrasesTermsParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of PhrasesTerms on a document.

    Attributes:

        sentences: list of sentences with PhrasesTerms extractions.
    """

    sentences: List[PhrasesTermsSentenceResult]


class PhrasesTermsPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of PhrasesTerms on a document.

    Attributes:

        paragraphs: list of paragraphs with PhrasesTerms extractions.
    """

    paragraphs: Union[List[PhrasesTermsParagraphResult], List[PhrasesTermsParagraphSentences]]


class PhrasesTermsPages(BaseModel):
    """
    Model that represents the list of pages of PhrasesTerms on a document.

    Attributes:

        version: version of the PhrasesTerms service used.
        pages_text: list of pages with PhrasesTerms extractions.
    """

    version: str
    pages_text: Union[List[PhrasesTermsPageResult], List[PhrasesTermsPageParagraphs]]


class PhrasesTermsDocumentDTO(BaseModel):
    """
    Model that contains PhrasesTerms data implemented in sentence data.

    Attributes:

            extractor_phrases_terms: The same structure with document.

    """

    extractor_phrases_terms: Union[PhrasesTermsDocument, PhrasesTermsPages]


class PhrasesWordbagDocumentInput(BaseDocumentInput):
    """
    Input model to extract PhrasesWordbag

    Attributes:

        language: object SDULanguage.
        result_output: Type of output format.
    """

    language: SDULanguage = SDULanguage(code="en", lang="english")
    result_output: ResultType = ResultType.pages


class PhrasesWordbagDocument(BaseModel):
    """
    Model that represents the result of named entity recognition PhrasesWordbag on a document for all doc.

    Attributes:

        version: version of the text extraction service used.
        result: PhrasesWordBagDTO object.
    """

    version: str
    result: PhrasesWordBagDTO


class PhrasesWordbagPageResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesWordbag on a document for page.

    Attributes:

        result: PhrasesWordBagDTO object.
    """

    result: PhrasesWordBagDTO


class PhrasesWordbagParagraphResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesWordbag on a document for paragraph.

    Attributes:

        result: PhrasesWordBagDTO object.
    """

    result: PhrasesWordBagDTO


class PhrasesWordbagSentenceResult(NestingId):
    """
    Model that represents the result of named entity recognition PhrasesWordbag on a document for sentence.

    Attributes:

        result: PhrasesWordBagDTO object.
    """

    result: PhrasesWordBagDTO


class PhrasesWordbagParagraphSentences(NestingId):
    """
    Model that represents the list of sentences of PhrasesWordbag on a document.

    Attributes:

        sentences: list of sentences with PhrasesWordbag extractions.
    """

    sentences: List[PhrasesWordbagSentenceResult]


class PhrasesWordbagPageParagraphs(NestingId):
    """
    Model that represents the list of paragraphs of PhrasesWordbag on a document.

    Attributes:

        paragraphs: list of paragraphs with PhrasesWordbag extractions.
    """

    paragraphs: Union[List[PhrasesWordbagParagraphResult], List[PhrasesWordbagParagraphSentences]]


class PhrasesWordbagPages(BaseModel):
    """
    Model that represents the list of pages of PhrasesWordbag on a document.

    Attributes:

        version: version of the PhrasesWordbag service used.
        pages_text: list of pages with PhrasesWordbag extractions.
    """

    version: str
    pages_text: Union[List[PhrasesWordbagPageResult], List[PhrasesWordbagPageParagraphs]]


class PhrasesWordbagDocumentDTO(BaseModel):
    """
    Model that contains PhrasesWordbag data implemented in sentence data.

    Attributes:

            extractor_phrases_wordbag: The same structure with document.

    """

    extractor_phrases_wordbag: Union[PhrasesWordbagDocument, PhrasesWordbagPages]


class RemoveFolderInputModel(BaseModel):
    """
    Input model for '/remove/folder' endpoint

    Parameters:

        data: A list of dictionaries. Each dictionary should contain 'collection_name',
              'uuid', and 'folder' keys. 'folder' key with the value being the path to the folder to be removed.
        return_only_successful: If True - only items for which the directory was successfully deleted will
                                be included in the response. If False, all items.

    """

    data: List[Dict]
    return_only_successful: bool


class ClearOutDocumentInputModel(BaseModel):
    """
    Input data model for "clear-out" router.

    Attributes:

        subdomain: tenant identifier
        client_id: client identifier
        days: how old documents to look for. Default: 3
        return_only_successful: If True - only items for which the directory was successfully deleted will be included in the response. If False, all items.
                                Default: False
        params: additional parameters for search
    """

    subdomain: str
    client_id: Optional[str] = None
    days: int = 3
    return_only_successful: bool = False
    params: Dict = {}


class BaseEmailCred(BaseModel):
    """
    Base email configuration model.

    Attributes:

        host: Email host. For example, "smtp.gmail.com"
        port: The port to use for the email service. For example, 587 for Gmail's SMTP.
    """

    host: str
    port: int


class UserCred(BaseModel):
    """
    User email credentials model.

    Attributes:

        email: Email address of the user.
        password: Password for the email account.
    """

    email: str
    password: str


class UserData(BaseModel):
    """
    User data model.

    Attributes:

        subdomain: Specific user subdomain.
        client_id: client ID. Default: "616fbefe-c6a0-489b-a942-b2098e46a3e2"
    """

    subdomain: str
    client_id: str = "616fbefe-c6a0-489b-a942-b2098e46a3e2"


class GmailCred(UserCred, BaseEmailCred):
    """
    Gmail specific credentials model.
    """

    pass


class BaseSenderConfig(BaseModel):
    """
    Base sender configuration model.

    Attributes:

        email_to: Recipient's email address.
        subject: The subject of the email. Default: "Subject".
        file_paths: List of file paths to be attached to the email.
        body: The body text of the email.
    """

    email_to: str
    subject: str = "Subject"
    file_paths: Optional[List[str]]
    body: str


class SenderConfigOutlook(BaseSenderConfig):
    """
    Sender configuration for Outlook.

    Attributes:

        office_tenant_id: The tenant ID for the Office 365 account.
        client_secret: Client secret for the Office 365 account.
        office_client_id: The client ID for the Office 365 account.
    """

    office_tenant_id: str
    client_secret: str
    office_client_id: str


class SenderConfigGmail(BaseEmailCred, BaseSenderConfig):
    """
    Sender configuration for Gmail.
    """

    pass


class SendEmailOutlook(BaseModel):
    """
    Outlook specific sender email parameters.

    Attributes:

        email: The email address of the sender.
        config: Configuration parameters for sending email through Outlook.
    """

    email: str
    config: SenderConfigOutlook


class SendEmailGmail(UserCred):
    """
    Gmail specific sender email parameters.

    Attributes:

        config: Configuration parameters for sending email through Gmail.
    """

    config: SenderConfigGmail


class ReadEmailGmail(UserData):
    """
    Gmail specific email reading parameters.

    Attributes:

        config: Gmail specific email reading configuration.
    """

    config: GmailCred


class OutlookCred(BaseModel):
    """
    Outlook credential model.

    Attributes:

        host: IMAP Outlook host. Default: "outlook.office365.com"
        client_id: The application's client ID.
        authority: The authority URL.
        secret: Secret key for the application.
        scope: Scopes for the application.
        email: Email for reading.
    """

    host: str = "outlook.office365.com"
    client_id: str
    authority: str
    secret: str
    scope: List[str]
    email: str


class ReaderEmailOutlook(UserData):
    """
    Outlook specific email reading parameters.

    Attributes:

        config: Outlook specific email reading configuration.
    """

    config: OutlookCred


class ConvertToHTMLInputModel(BaseModel):
    """
    Input data model for "convert-to-html" router.

    Attributes:

        full_file_path: the path to the file to be converted
        document_id: document identifier
        from_pandas: set true if need html from pandas
                    Default from apache tika
    """

    full_file_path: str
    document_id: Optional[str] = None
    from_pandas: bool = False


class ConvertToHTMLOutputModel(BaseModel):
    """
    Output data model for "convert-to-html" router.

    Attributes:

        document_id: document identifier
        data: HTMLConverterResponse object
    """

    document_id: Optional[str] = None
    data: HTMLConverterResponse


class FileType(str, Enum):
    XLS = ".xls"
    XLSX = ".xlsx"
    CSV = ".csv"
    XML = ".xml"
    JSON = ".json"


class MailType(str, Enum):
    EML = ".eml"
    MSG = ".msg"


class ConverterEmailInputModel(BaseModel):
    """
    Input model for 'parse-email-file' router

    Attributes:

        full_file_path: the path to the file to be converted
    """

    full_file_path: str


class ConvertToTextInputModel(BaseModel):
    """
    Input data model for "convert-to-text" router.

    Attributes:

        full_file_path: the path to the file to be converted.
    """

    full_file_path: str


class HTMLToPDFInputModel(BaseModel):
    """
    Input data model for "html-to-pdf" router.

    Attributes:

        str_html: html as str
        path_to_save: path to save with filename. Ex: data/filename.pdf
                    if not set, it will return a string of bytes
    """

    str_html: str
    path_to_save: str = None


class OCRConfig(BaseModel):
    """
    OCR config

    Attributes:

        ocr_clean: boolean flag to indicate whether to apply OCR cleaning or not. Default: True
        ocr_deskew: boolean flag to indicate whether to deskew the image before OCR or not. Default: True
        ocr_tqm_progress: boolean flag to disable or enable progress bar. Default: True
        ocr_image_dpi:  the DPI of the input image. Default 300
        ocr_languages: list of languages to be used for OCR. Default: ["eng", "deu", "fra", "spa", "ita"]
        ocr_optimize: flag to indicate whether to optimize OCR. Defaults: 1
        ocr_page_seqmode: allows you to pass page segmentation arguments to Tesseract OCR. Default: 1
        option_embedded_ocr: flag to indicate if the OCR should be done for embedded images. Default: False
        option_skip_text: skip pages with text. Default: True

    """

    ocr_clean: bool = True
    ocr_deskew: bool = True
    ocr_tqm_progress: bool = True
    ocr_image_dpi: int = 300
    ocr_languages: List = ["eng", "deu", "fra", "spa", "ita"]
    ocr_optimize: int = 1
    ocr_page_seqmode: int = 1
    option_embedded_ocr: bool = False
    option_skip_text: bool = True


class PDFConverterResult(BaseModel):
    """
    A class to represent the result of a PDF conversion operation.

    Attributes:

        metadata: dictionary containing metadata information of the PDF file.
        html_path: path to file with HTML representation of the PDF.
        images: list of `SDUPageImage` objects representing images.
        pages_layout: list of `SDULayout` objects representing the layout of each page in the PDF.
        pages_text: list of `SDUPage` objects representing the text content of each page in the PDF.
        full_file_path: full path of the input PDF file.
        debug_file_path: full path of the debug file.
        readorder_file_path: full path of the read order file.
        clean_text_path: path to file with clean text content of the PDF.
        lang: `SDULanguage` object representing the detected language of the PDF.
        raw_text_path: path to file with raw text content of the PDF.
    """

    metadata: Dict = {}
    html_path: str = ""
    images: List[SDUPageImage] = []
    pages_layout: List[SDULayout] = []
    pages_text: List[SDUPage] = []
    full_file_path: str = ""
    debug_file_path: str = ""
    readorder_file_path: str = ""
    clean_text_path: str = ""
    lang: SDULanguage = SDULanguage()
    raw_text_path: str = ""


class BBFormat(Enum):
    """
    Enumeration of bounding box formats.

    Attributes:

        XYWH: indicates the bounding box format as (x, y, width, height).
        XYX2Y2: indicates the bounding box format as (x1, y1, x2, y2).

    """

    XYWH = 1
    XYX2Y2 = 2


class CreatePDFInputModel(BaseModel):
    """
    Input data model for "create-pdf" router.

    Attributes:

        full_file_path: full file path of the input file.
        document_id: optional document ID.
    """

    full_file_path: str
    document_id: Optional[str] = None


class CreatePDFOutputModel(BaseModel):
    """
    Output data model for "create-pdf" router.

    Attributes:

        data: full file path of the input file.
        document_id: optional document ID.
    """

    data: PDFConverterResult
    document_id: Optional[str] = None


class TemplateContent(BaseModel):
    """
    Model contains template content

    Attributes:

        main_tpl_code: code for main template
        pdf_tpl_code: code for template to convert to pdf
    """

    main_tpl_code: str
    pdf_tpl_code: Optional[str] = ""


class OutputType(str, Enum):
    """
    Possible variants of output type for EngineTemplate
    """

    PDF = "pdf"
    HTML = "html"


class TemplateInput(BaseModel):
    """
    Input model

    Attributes:

        output_type: type of output file. Default: pdf
        template_name: name of template
        doc_data: variables for filling template
        template_version: template version. Default: v1
        tenant_id: tenant identifier
        document_id: document identifier
        template_content: content for templates
    """

    output_type: OutputType = OutputType.PDF
    template_name: str
    doc_data: Dict = {}
    template_version: str = "v1"
    tenant_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    template_content: Optional[TemplateContent] = {}


class PublishInputModel(BaseModel):
    """
    Input data model for "publish" router.

    Attributes:

        message: text to publish
        topic_name: pubsub topic name
        service_name: service that publishes the message
    """

    message: str
    topic_name: str
    service_name: Optional[str] = None


class BarcodeInput(BaseModel):
    """
    Input model

    Attributes:

        client_id: collection name(user identifier)
        subdomain: tenant identifier
        document_id: document identifier
    """

    document_id: str
    subdomain: str
    client_id: str


class BarcodeDTO(BaseModel):
    """
    Output model

    Attributes:

        result: Union[Dict, List].
    """

    result: Union[Dict, List]


class CreatePDFInputModel(BaseModel):
    """
    Input data model for "create-pdf" router.

    Attributes:

        full_file_path: full file path of the input file.
        document_id: optional document ID.
    """

    full_file_path: str
    document_id: Optional[str] = None
