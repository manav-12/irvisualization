python -m backend.pylucene_retriever.IndexFiles
python -m backend.core_ir.DocIndexer
python -m backend.utils.processData
python -m backend.expansion.Word2Vec
python -m backend.utils.dataUtils
python -m backend.utils.createStemWordDict