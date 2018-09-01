import pytest
from unittest.mock import MagicMock
from indexing.exampleindex import *
import ainix_common.parsing.parseast
from indexing.whooshbackend import WhooshIndexBackend

def test_nearest_example():
    type_context = TypeContext()
    type_name = "DeepThoughts"
    other_type_name = "PythonThoughts"
    top_type = AInixType(type_context, type_name)
    other_type = AInixType(type_context, other_type_name)
    parsed_rep = ainix_common.parsing.parseast.indexable_repr_classify_type(top_type.name)
    other_parsed_rep = ainix_common.parsing.parseast.indexable_repr_classify_type(other_type.name)

    index = ExamplesIndex(type_context, backend=ExamplesIndex.get_default_ram_backend())
    index.add_example(Example("what is the meaning of life", "42",
                              index.DEFAULT_X_TYPE, type_name, 1, parsed_rep))
    index.add_example(Example("what is the meaning of the universe", "forty two",
                              index.DEFAULT_X_TYPE, type_name, 1, parsed_rep))
    index.add_example(Example("what... is the airspeed of an unladen swallow?",
                              "African or European?", index.DEFAULT_X_TYPE,
                              other_type_name, 1, other_parsed_rep))
    index.add_example(Example("what is the meaning of everything else", "four two",
                              index.DEFAULT_X_TYPE, type_name, 1, parsed_rep))
    index.add_example(Example("what... is your quest?",
                              "To seek the Holy Grail", index.DEFAULT_X_TYPE,
                              other_type_name, 1, other_parsed_rep))
    result = index.get_nearest_examples("what... is your quest?", other_type.name)
    assert len(result) == 2
    assert result[0].xquery == "what... is your quest?"

    result = index.get_nearest_examples("what... is your quest?", other_type.name, max_results=1)
    assert len(result) == 1
    assert result[0].xquery == "what... is your quest?"
    result = index.get_nearest_examples("everything else", top_type.name, max_results=1)
    assert len(result) == 1
    assert result[0].xquery == "what is the meaning of everything else"
