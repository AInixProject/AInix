import indexing.exampleindex
from indexing import index, exampleloader
from ainix_common.parsing.typecontext import TypeContext
from ainix_common.parsing import loader

type_context = TypeContext()
loader.load_path("../builtin_types/numbers.ainix.yaml", type_context)
loader.load_path("../builtin_types/generic_parsers.ainix.yaml", type_context)
type_context.fill_default_parsers()

index = indexing.exampleindex.ExamplesIndex(type_context)
exampleloader.load_path("../builtin_types/numbers_examples.ainix.yaml", index)

print("doc count", index.backend.index.doc_count())
r = index.get_nearest_examples("five", "Number")
for result in r:
    print(result)

from ainix_common.parsing.parseast import StringParser

real_parser = StringParser(type_context.get_type_by_name("Number"))
actual = real_parser.create_parse_tree("5")

from models.SeaCR.seacr import SeaCRModel

model = SeaCRModel(index)

print("predict")
prediction = model.predict("five", "Number")
print("predicted")
print(prediction.dump_str())
print("actual")
print(actual.dump_str())



