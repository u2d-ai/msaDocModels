# msaDocModels Release Notes
## Possible future features:

# 0.0.103

- fix Barcode model

# 0.0.102

- add models for ConvertorSeparation and models for document Taxonomy 

# 0.0.100

- add Input and DTO models for Spellcheck 

# 0.0.98

- fix algorithm to DocClassifier Input 

# 0.0.97

- add algorithm to DocClassifier Input 
- change dto model for DocClassifier

# 0.0.96

- add include_fields field to CreatePDFInputModel

# 0.0.95

- add Enum with Keywords algorithms

# 0.0.94

- enum instead Literal for TemplateInput model

# 0.0.93

- added email_tags field to EmailConverterResponse model

# 0.0.92

- added default value for tenant_id and document_id in TemplateInput model

# 0.0.91

- added optional tenant_id and document_id for TemplateInput model

# 0.0.90

- remove docx from model, added optional document_id, fix Barcode models

# 0.0.89

- changed model, learnset, testset models

# 0.0.88

- add field embedding_attachments to separate attachments

# 0.0.87

- add patterns for models TextExtractionFormatsInput
- separate DTO model for three TextExtractionFormatsStrDTO, TextExtractionFormatsListDTO, TextExtractionFormatsDictDTO

# 0.0.86

- fix model queue in TextExtractionFormatsDTO

# 0.0.85

- fix EntityExtractorInput, EntityExtractorDocumentInput models

# 0.0.84

- update models for TemplateInput, EntityExtractorInput, EntityExtractorDocumentInput
- add models for work with database updates from pub/sub

## 0.0.83

- Added model with optional attachments in converteremail svc

# 0.0.82

- fix model

# 0.0.81

- added new models

# 0.0.80

- fix DBBaseDocumentInput
- added ConvertToZIPInput, ConvertToXLSXInput

# 0.0.79

- fix ProcessStatus model

# 0.0.78

- added RemoveFolderInputModel and ClearOutDocumentInputModel models

# 0.0.77

- change initial status to document

# 0.0.76

- add project to filter document by status

# 0.0.75

- add models for working with document statuses

# 0.0.74

- add model to working with document for Summary/Phrases

# 0.0.73

- add output_file_paths field to document
- fix model to working with documents

# 0.0.72

- add models to working with full document/pages/paragraphs/sentences

# 0.0.71

- add data field to store files data
- add project per document
- change document status to string

# 0.0.70

- fix model TextExtractionDefaultsDTO

# 0.0.69

- fix information_extraction_answer key

# 0.0.68

- Change keys from answers/questions to result

# 0.0.67

- Add models for information extraction

# 0.0.66

- Remove taxonomy stuff

## 0.0.65

- Remove fields active and inherited for learnsets/testsets/models
- Add models to get list of learnsets/testsets/models

## 0.0.64

- Add models to extended endpoints in summary

## 0.0.63

- Add DTO models to extract phrases from text

## 0.0.62

- Add models to extract phrases from text

## 0.0.61

- Add models to extract keywords from text

## 0.0.60

- Add default variables for Phrases mining

## 0.0.59

- Add models for Phrases mining

## 0.0.58

- Changed names of models to Phrases word bag
- Fix MSAOpenAPIInfo model


## 0.0.57

- Add possible to save learnset object and testset object.

## 0.0.56

- remove language from extraction NLP

## 0.0.55

- change model for NotaryDTO, when notary are not found

## 0.0.54

- add input and output  models for Notary

## 0.0.53

- fix NER results model

## 0.0.52

- fix Defaults results model

## 0.0.51

- clean document structure, add models for NLP, NER, Defaults results

## 0.0.50

- fix fields for TextExtractionDocumentNLPInput

## 0.0.49

- add defaults for TextExtractionDefaults model

## 0.0.48

- add id to ExtractionDefaultResult, RecognizerDefaultResult

## 0.0.47

- add document_id models for TextExtractionDocumentNLPInput

## 0.0.46

- fix document_id models for EntityExtractorDocumentDTO

## 0.0.45

- add document_id models for EntityExtractorDocumentDTO

## 0.0.44

- fix models for TextExtractionDefaults

## 0.0.43

- add models for TextExtractionDefaults, TextExtractionNLP

## 0.0.42

- change structure for DBBaseDocumentInput

## 0.0.41

- add model_data to AutoMLStatus model

## 0.0.40

- fix typo in webhook url

## 0.0.39

- add webhook url and constant to train model

## 0.0.38

- add language for DataCleanAIInput

## 0.0.37

- change case for default Language

## 0.0.36

- add language for ExtractKeywordsInput

## 0.0.35

- Change variables for DataCleanAIDTO

## 0.0.34

- Change structure for clean text for model

## 0.0.33

- Change structure for clean text for model
- Add more documentation
- Add models for extract keywords

## 0.0.32

- Change model for build ai model
- Add models for clean text for model

## 0.0.31

- Pinned package versions

## 0.0.30

- Change model for inference, can allow list of target columns

## 0.0.29

- Change model for convert xlsx file

## 0.0.28

- Change input/output models to Profiling

## 0.0.27

- Add model ConversionInput

## 0.0.26

- Add field content_unzipped_files to EmailConverterResponse

## 0.0.25

- Add model of documents ids, change all document_id to optional

## 0.0.24

- Change document_id to optional, add process to document status

## 0.0.23

- Added  models to working with DBLayer, HTMLconverter, EmailConverter


## 0.0.12

- Added  models to working with Text, Language/Statistics/Segmentation/Phrases/Summary/Sentiment

## 0.0.2

- Added msg module, allows generic API JSON message creation with capabilities to re-create original datatypes and class instances

## 0.0.1

- This is the first public release of msaDocModels, former releases are all stages of development and internal releases.

