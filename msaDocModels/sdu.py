import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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


class RecognizerDefaultResult(ExtractionDefaultResult):
    """
    Model for representing a recognized entity.

    Attributes:
        type: The type of the recognized entity.
    """

    type: str


class TextExtractionDefaults(BaseModel):
    """
    Data transfer object for  entity extractor.

    Attributes:
        recognizer: List of recognizer results.
        zipcode: List of extracted zipcodes.
        phone_number: List of extracted phone numbers.
        url: List of extracted URLs.
        credit_card: List of extracted credit card numbers.
        credit_cards_strict: List of strictly extracted credit card numbers.
        ipv4: List of extracted IPv4 addresses.
        ipv6: List of extracted IPv6 addresses.
        mac_address: List of extracted MAC addresses.
        hex_value: List of extracted hexadecimal values.
        slug: List of extracted slugs.
        bitcoin_address: List of extracted Bitcoin addresses.
        yandex_money_address: List of extracted Yandex Money addresses.
        latitude: List of extracted latitudes.
        longitude:  List of extracted longitudes.
        irc :  list  IRCs
        license_plate : list License plates
        time : list Time
        iso_datetime : list ISO datetime
        isbn : list ISBNs
        roman_numeral : list Roman numerals
        ethereum_address : list Ethereum addresses
        ethereum_hash : list Ethereum hashes
        uuid : list UUIDs
        float_number : list Float numbers
        pgp_fingerprint : list PGP fingerprints
        pesel :list PESELs
    """

    recognizer: List[RecognizerDefaultResult] = []
    zipcode: List[ExtractionDefaultResult] = []
    phone_number: List[ExtractionDefaultResult] = []
    url: List[ExtractionDefaultResult] = []
    credit_card: List[ExtractionDefaultResult] = []
    credit_cards_strict: List[ExtractionDefaultResult] = []
    ipv4: List[ExtractionDefaultResult] = []
    ipv6: List[ExtractionDefaultResult] = []
    mac_address: List[ExtractionDefaultResult] = []
    hex_value: List[ExtractionDefaultResult] = []
    slug: List[ExtractionDefaultResult] = []
    bitcoin_address: List[ExtractionDefaultResult] = []
    yandex_money_address: List[ExtractionDefaultResult] = []
    latitude: List[ExtractionDefaultResult] = []
    longitude: List[ExtractionDefaultResult] = []
    irc: List[ExtractionDefaultResult] = []
    license_plate: List[ExtractionDefaultResult] = []
    time: List[ExtractionDefaultResult] = []
    iso_datetime: List[ExtractionDefaultResult] = []
    isbn: List[ExtractionDefaultResult] = []
    roman_numeral: List[ExtractionDefaultResult] = []
    ethereum_address: List[ExtractionDefaultResult] = []
    ethereum_hash: List[ExtractionDefaultResult] = []
    uuid: List[ExtractionDefaultResult] = []
    float_number: List[ExtractionDefaultResult] = []
    pgp_fingerprint: List[ExtractionDefaultResult] = []
    pesel: List[ExtractionDefaultResult] = []


class TextExtractionDefaultsDTO(BaseModel):
    """DTO, representing the result of extraction defaults"""

    extractions: Union[
        TextExtractionDefaults,
        List[TextExtractionDefaults],
        Dict[Any, TextExtractionDefaults],
    ]


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
            If False, includes the city, address, first name, last name, and office title full. Defaults: False.

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
    Input model to detect Segmentation

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
    DTO, representing the result of service Keywords.

    Attributes:
        keywords:  List of keywords and/or keyphrases.
    """

    keywords: List[Union[List, List[Union[str, int]]]]


class ExtractKeywordsInput(BaseModel):
    """
    Data input model for ExtractKeywords.

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
    Data input model for ExtractKeywords.

    Attributes:
        algorithms: which algorithms use for extract. Can be list of ["yake", "bert", "bert_vectorized", "tf_idf"]
        language: default is german
    """
    algorithms: List[str] = ["yake", "bert"]
    language: SDULanguage = SDULanguage(code="de", lang="german")


class ExtractKeywordsTextDTO(BaseModel):
    """
    DTO, representing the result of service Keywords.

    Attributes:
        data: Extracted keywords for text.
    """
    data: Union[List[str], List[List[str]], Dict[Any, List[str]]]


class ExtractKeywordsDTO(BaseModel):
    """
    DTO, representing the result of service Keywords.

    Attributes:
        data: extended input text by InputKeyKeys, have the len as input.
    """

    data: List[Dict[str, Dict[str, Any]]]


class SummaryInput(DocumentLangInput):
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


class StatisticsInput(DocumentLangInput):
    """Data input model for Statistics."""


class StatisticsDTO(SDUStatistic):
    """DTO, representing the result of service Statistics."""


class SummaryEmbeddedInput(DocumentLangInput):
    """Data input model for Summary Embedded."""


class SentenceTopicsInput(DocumentLangInput):
    """
    Data input model for Sentence Topics.

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
    """DTO, representing the result of service Summary Embedded.
    Attributes:
        sentences_summary: List of sentences along with their respective rates.
    """
    sentences_summary: List[SentenceSummary]


class SummaryDTO(wdc.WDCItem):
    """DTO, representing the result of service Summary."""


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
    DTO, representing the result of service Taxonomy Cities.

    Attributes:

        cities: List of Cities.
        cities_winner: winner object City.
    """

    cities: List[City]
    cities_winner: Optional[City]


class TaxonomyCountriesDTO(BaseModel):
    """
    DTO, representing the result of service Taxonomy Countries.

    Attributes:

        countries: List of Countries.
        countries_winner: winner object Country.
    """

    countries: List[Country]
    countries_winner: Optional[Country]


class TaxonomyCompaniesDTO(BaseModel):
    """
    DTO, representing the result of service Taxonomy Companies.

    Attributes:

        companies: List of Companies.
        companies_winner: winner object Company.
    """

    companies: List[Company]
    companies_winner: Optional[Company]


class TaxonomyDTO(TaxonomyCountriesDTO, TaxonomyCompaniesDTO, TaxonomyCitiesDTO):
    """DTO, representing the result of service Taxonomy."""


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

    number: int = 0
    timestamp: str = str(datetime.utcnow())


class DBBaseDocumentInput(BaseModel):
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
        tags: list of tags.
        language: language.
        needs_update: need update or not.
        data: data.
        images: images.
        pages_layout: layouts.
        pages_text: pages.
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
    tags: Optional[Dict] = {}
    language: Optional[SDULanguage] = None
    needs_update: bool = False
    data: Optional[SDUData] = None
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


class UpdateAI(BaseModel):
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


class TaxonomyDataInput(BaseInfo):
    """
    AI taxonomy input.

    Attributes:

        taxonomies: list of taxonomies objects.
    """

    taxonomies: List[Dict]


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


class TaxonomyDataDTO(TaxonomyDataInput, MongoId):
    """
    AI taxonomy output.
    """


class ConversionInput(BaseModel):
    """
    Model that contains inference data along with filenames to use for XLSX conversion.

    Attributes:

        filenames: list of filenames that files should be saved as
        inference: inference data, first key means sheet name for XLSX file
    """

    filenames: List[str]
    inference: List[Dict[str, Dict[str, Any]]]


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
        txt_content: SDUText.
        msg: SDUEmail.
        content_unzipped_files: object of
    """

    content_attachments: List[SDUAttachment]
    txt_content: SDUText
    msg: SDUEmail
    content_unzipped_files: Optional[List[HTMLConverterResponse]]


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


class EntityExtractorInput(DocumentLangInput):
    """Model that contains input data for extract defaults."""


class EntityExtractorDocumentInput(BaseModel):
    """
    Model that contains input data for extract defaults.

    Attributes:

        pages_text: The document data.
        document_id: optional uuid for document.
    """

    pages_text: List[SDUPage] = []
    document_id: Optional[UUID4]


class TextExtractionNLPInput(DocumentInput):
    """
    Data input model for extraction NLP from text.
    """


class TextExtractionDocumentNLPInput(BaseModel):
    """
    Data input model for extraction NLP from document.

    Attributes:

        pages_text: The document data.
        document_id: optional uuid for document.
    """

    pages_text: List[SDUPage] = []
    document_id: Optional[UUID4]


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
    Data input model for Text Clean.

    Attributes:

        extractions: List of ExtractionNLP.
    """

    extractions: Union[List[ExtractionNLP], List[List[ExtractionNLP]], Dict[Any, List[ExtractionNLP]]]


class TextExtractionDocumentDefaultsDTO(BaseModel):
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


class TextExtractionNLPDocumentDTO(BaseModel):
    """
    Model that contains nlp data implemented in sentence data.

    Attributes:

        text_extraction_nlp: The same structure with document.

    """

    text_extraction_nlp: TextExtractionDocumentNLPPage


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


class TextExtractionDocumentNERInput(BaseModel):
    """
    Model that represents an input for named entity recognition text extraction on a document.

    Attributes:

        pages_text: list of pages to perform named entity recognition on.
    """

    pages_text: List[SDUPage] = []
    document_id: Optional[UUID4]


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


class TextExtractionNERDocumentDTO(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        text_extraction_ner: result of named entity recognition text extraction on a document
                            using the TextExtractionService.
    """

    text_extraction_ner: TextExtractionDocumentNERDTO


class SentenceDefaultsDTO(NestingId):
    """
    Model that represents a sentences with NLP extractions.

    Attributes:

        result: list of sentences with nlp found in the page.
    """

    result: TextExtractionDefaults


class ParagraphDefaultsDTO(NestingId):
    """
    Model that represents a paragraph with NLP extractions.

    Attributes:

        sentences: list of sentences.
    """

    sentences: List[SentenceDefaultsDTO] = []


class PageDefaultsDTO(NestingId):
    """
    Model that represents a page with named entity recognition extractions.

    Attributes:

        paragraphs: list of paragraphs.
    """

    paragraphs: List[ParagraphDefaultsDTO] = []


class TextExtractionDocumentDefaultsPage(BaseModel):
    """
    Model that represents the result of named entity recognition text extraction on a document.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with NLP extractions.
    """

    version: str
    pages_text: List[PageDefaultsDTO] = []


class TextExtractionDefaultsDocumentDTO(BaseModel):
    """
    Model that contains nlp data implemented in sentence data.

    Attributes:

            text_extraction_defaults: The same structure with document.

    """

    text_extraction_defaults: TextExtractionDocumentDefaultsPage


class PageNotaryDTO(NestingId):
    """
    Model that represents a page with notary extractions.

    Attributes:

        result: Notary object if found notary or empty dict.
    """

    result: Union[Notary, Dict] = {}


class TextExtractionNotaryDocumentPage(BaseModel):
    """
    Model that represents the result of search notary in text.

    Attributes:

        version: version of the text extraction service used.
        pages_text: list of pages with Notary extractions.
    """

    version: str
    pages_text: List[PageNotaryDTO] = []


class TextExtractionNotaryDocumentDTO(BaseModel):
    """
    Model that contains notary data implemented in page data.

    Attributes:

            text_extraction_notary: The same structure with document.

    """

    text_extraction_notary: TextExtractionNotaryDocumentPage
