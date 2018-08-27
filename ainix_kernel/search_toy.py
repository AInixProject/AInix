from indexing import index, exampleloader
from ainix_common.parsing.typecontext import TypeContext
from ainix_common.parsing import loader

type_context = TypeContext()
loader.load_path("../builtin_types/numbers.ainix.yaml", type_context)
loader.load_path("../builtin_types/generic_parsers.ainix.yaml", type_context)
type_context.fill_default_parsers()

index = index.ExamplesIndex(type_context)
exampleloader.load_path("../builtin_types/numbers_examples.ainix.yaml", index)

print("doc count", index.backend.index.doc_count())
r = index.get_nearest_examples("five", y_type="Number")
for result in r:
    print(result)
