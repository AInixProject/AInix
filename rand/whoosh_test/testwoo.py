from whoosh.index import create_in
from whoosh.fields import *
schema = Schema(cmd=TEXT(stored=True), nl=TEXT(stored=True))

ix = create_in("indexdir", schema)
writer = ix.writer()
writer.add_document(nl=u"First document",
                    cmd=u"This is the first document we've added!")
writer.add_document(nl=u"Second document",
                    cmd=u"The second one is even more interesting!")
writer.commit()
#from whoosh.qparser import QueryParser
from whoosh.query import And, Term
with ix.searcher() as searcher:
    myquery = And([Term("cmd", u"document")])
    #query = QueryParser("content", ix.schema).parse("first")
    results = searcher.search(myquery)
    print(results)
    for result in results:
        print(result)

