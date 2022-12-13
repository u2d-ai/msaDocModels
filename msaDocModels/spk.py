from typing import Dict, List, Optional, Union

from pydantic import UUID4, BaseModel

from msaDocModels import sdu, wdc
from msaDocModels.sdu import SDULanguage


class TenantIdInput(BaseModel):
    """Input model with tenant_id"""

    tenant_id: UUID4


class TextInput(BaseModel):
    """Input model with input_text"""

    input_text: Union[str, List, Dict]


class DocumentInput(TextInput):
    """Input document model"""

    document_id: UUID4


class SentencesInput(BaseModel):
    """Input model with sentences"""

    document_id: UUID4
    sentences: List[str]


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

    document_id: UUID4
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
