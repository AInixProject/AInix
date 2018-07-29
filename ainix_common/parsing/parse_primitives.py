from collections import namedtuple


class AInixType:
    """Used to contruct AInix types.

    Types represent a collection of related objects.

    Args:
        type_name : An upper CamelCase string to identify this type
    """
    def __init__(self, name : str, default_value_parser = None):
        self.name = name
        self.default_parser = default_value_parser


class AInixObject:
    def __init__(
        self,
        name: str,
        type: AInixType,
        children: list,
        direct_sibling: 'AInixArgument' = None
    ):
        self.name = name
        self.type = type
        self.children = children
        self.direct_sibling = direct_sibling


class ValueParser:
    def __init__(self):
        pass


class ObjectParser:
    def __init__(self, object):
        self.object = object

    def parse_string(self, string: str) -> 'ParserResult':
        result = ParserResult(self.object, string)
        self._parse_string(string, result)
        return result

    def _parse_string(self, string: str, result: 'ParserResult'):
        pass


class ParserResult:
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

class AInixArgument:
    def __init__(
        self,
        name: str,
        type: AInixType,
        value_parser: ValueParser = None,
        required: bool = False,
        parse_data: dict = {}
    ):
        self.name = name
        self.type = type
        self.required = required
        if ValueParser is not None:
            self.value_parser = value_parser
        else:
            if type.default_parser is None:
                raise ValueError("No value_parser provided for an AInixArgument. \
                    However, type %s does not provide a default value_parser"
                    % (self.type.name,))
            self.value_parser = type.value_parser
        self.parse_data = parse_data
