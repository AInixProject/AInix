from collections import namedtuple
from typing import Type


class TypeParser:
    def __init__(self, type):
        self.type = type
        type_graph = type.type_graph
        self.type_implementations = type_graph.get_implementations(type)

    @staticmethod
    def _match_attribute(object_list, key, value):
        return [o for o in object_list if o.type_data[key] == value]

    def parse_string(self, string: str) -> 'TypeParserResult':
        result = TypeParserResult(self.type, string)
        self._parse_string(string, result)
        return result

    def _parse_string(self, string: str, result):
        raise NotImplementedError()


class TypeParserResult:
    def __init__(self, type, string):
        self.type = type
        self.string = string
        self._implementation = None
        self._next_slice = None
        self._next_parser: Type[ObjectParser] = type.default_object_parser

    def get_implementation(self):
        return self._implementation

    def get_next_string(self):
        si, ei = self._next_slice
        return self.string[si:ei].strip()

    def set_valid_implementation(self, implementation):
        self._implementation = implementation

    def set_next_slice(self, start_idx, end_idx):
        self._next_slice = (int(start_idx), int(end_idx))

    @property
    def next_parser(self) -> 'ObjectParser':
        return self._next_parser(self._implementation)


class ObjectParser:
    def __init__(self, object):
        self.object = object

    def parse_string(self, string: str) -> 'ObjectParserResult':
        result = ObjectParserResult(self.object, string)
        self._parse_string(string, result)
        return result

    def _parse_string(self, string: str, result: 'ObjectParserResult'):
        raise NotImplementedError()


class ObjectParserResult:
    def __init__(self, object, string):
        self._object = object
        self._result_dict = {}
        self._sibling_result = None
        self.string = string
        self.ArgData = namedtuple("PresentData", ['slice', 'slice_string'])

    def _get_slice_string(self, start_idx: int, end_idx: int):
        return self.string[start_idx:end_idx].strip()

    def get_arg_present(self, name):
        return self._result_dict.get(name, None)

    def get_sibling_arg(self):
        return self._sibling_result

    def set_arg_present(self, arg_name: str, start_idx, end_idx):
        # TODO (DNGros): check that the argname exists
        si, ei = int(start_idx), int(end_idx)
        self._result_dict[arg_name] = self.ArgData(
            (si, ei), self._get_slice_string(si, ei))

    def set_sibling_present(self, start_idx, end_idx):
        si, ei = int(start_idx), int(end_idx)
        self._sibling_result = self.ArgData(
            (si, ei), self._get_slice_string(si, ei))


class AInixParseError(Exception):
    pass


class SingleTypeImplParser(TypeParser):
    def _parse_string(self, string: str, result):
        num_impls = len(self.type_implementations)
        if num_impls != 1:
            raise AInixParseError("{self.__class__.name} expects exactly one implementation. "
                                  "Found {num_impls} implementations.")

        result.set_valid_implementation(self.type_implementations[0])
        result.set_next_slice(0, len(string))


