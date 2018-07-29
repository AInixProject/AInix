from collections import namedtuple



class ValueParser:
    def __init__(self):
        pass


class ObjectParser:
    def __init__(self, object):
        self.object = object

    def parse_string(self, string: str) -> 'ObjectParserResult':
        result = ObjectParserResult(self.object, string)
        self._parse_string(string, result)
        return result

    def _parse_string(self, string: str, result: 'ObjectParserResult'):
        pass


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

