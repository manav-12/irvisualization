INDEX_DIR = "IndexFiles.index"

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions, Term
from org.apache.lucene.store import SimpleFSDirectory


class LuceneIndexUpdater:

    def __init__(self):
        vm_env = lucene.getVMEnv()
        vm_env.attachCurrentThread()
        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setTokenized(False)
        self.t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        self.t2 = FieldType()
        self.t2.setStored(False)
        self.t2.setTokenized(True)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        self.store = SimpleFSDirectory(Paths.get('backend/data/files_lucene'))
        self.analyzer = LimitTokenCountAnalyzer(StandardAnalyzer(), 1048576)

    def update_lucene_q_idx(self, doc_id, doc_text):
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.APPEND)
        writer = IndexWriter(self.store, config)
        try:
            doc = Document()
            doc.add(Field("name", doc_id, self.t1))
            if len(doc_text) > 0:
                doc.add(Field("contents", doc_text, self.t2))
            else:
                print("warning: no content in %s", doc_id)
                doc.add(Field("contents", '', self.t2))
            writer.addDocument(doc)
        except:
            print("Error occurred")

        writer.commit()
        writer.close()

    def delete_q_from_idx(self):
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.APPEND)
        writer = IndexWriter(self.store, config)
        writer.deleteDocuments(Term('name', "Query"))
        writer.commit()
        writer.close()


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    ls_update = LuceneIndexUpdater()
    ls_update.delete_q_from_idx()
