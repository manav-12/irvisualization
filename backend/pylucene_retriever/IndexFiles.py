#!/usr/bin/env python
import operator

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime
import csv
import glob, re
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from backend.pylucene_retriever.PorterStemmerAnalyzer import PorterStemmerAnalyzer

"""
This class is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""
csv.field_size_limit(10000000)


class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


class IndexFiles(object):
    def __init__(self, data_flag, analyzer):
        store_dir = os.path.join("backend", "data", data_flag, "files_lucene")
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        store = SimpleFSDirectory(Paths.get(store_dir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # with open('backend/files_lucene/exclude_docs.txt', 'r') as f:
        #      self.doc_names = f.read().splitlines()
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)
        self.indexDocs(data_flag, writer)
        ticker = Ticker()
        print('commit index', )
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    def indexDocs(self, dataflag, writer):
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        t2 = FieldType()
        t2.setStored(False)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        print("test marker")
        data_path = os.path.join("backend", "data", data_flag, "data_csvs", "SNOW_" + data_flag + ".csv")
        with open(data_path, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            next(reader)
            sorted_doc = {}
            for row in reader:
                if row[1].strip() != '':
                    sorted_doc[row[0]] = row[1]
        print("data loaded")
        for filename, content in sorted_doc.items():
            print("adding", filename)
            try:
                doc = Document()
                if dataflag == 'TOTALRECALL':
                    file_name = "{:06d}".format(int(filename))
                    doc.add(Field("name", file_name, t1))
                else:
                    doc.add(Field("name", filename, t1))

                if len(content) > 0:
                    doc.add(Field("contents", content, t2))
                else:
                    print("warning: no content in %s" % filename)
                    doc.add(Field("contents", '', t2))
                writer.addDocument(doc)
            except Exception as e:
                print("Failed in indexDocs:")


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    start = datetime.now()
    try:
        data_flag = ''  # ROBUST
        IndexFiles(data_flag, StandardAnalyzer())
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Faileds: ", e)
