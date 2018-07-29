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

    def parse_string(self, string: str):
        result = ParserResult(self.object)
        self._parse_string(self, string, result)
        return result

    def _parse_string(self, string:str, result: 'ParserResult'):
        pass


class ParserResult:
    def __init__(self, object):
        self._object = object
        self._result_dict = {}
        self._sibling_result = None

    def set_arg_present(self, arg_name: str, start_idx, end_idx):
        # TODO (DNGros): check that the argname exists
        self._result_dict[arg_name] = (start_idx, end_idx)

    def set_sibling_present(self, start_idx, end_idx):
        self._sibling_result = (start_idx, end_idx)


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
