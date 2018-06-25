from data import *
import hjson
"""Some quick functions for converting the old hard coded examples to config version"""
from collections import defaultdict
from collections import OrderedDict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
def get_example_list(tuple_data):
    outlist = []
    cmdToDescs = defaultdict(list)
    for lang, cmd in tuple_data:
        cmdToDescs[cmd].append(lang)
    for cmd, langlist in cmdToDescs.items():
        outlist.append(OrderedDict([
            ("lang", langlist),
            ("cmd", [cmd])
        ]))
    outlist.sort(key = lambda x: (len(x["lang"]), -len(x["cmd"][0])), reverse = True)
    return outlist

def dump_desc(desc):
    print(desc)
    out = OrderedDict({
        "name": desc.name,
    })
    argsList = []
    for arg in desc.arguments:
        argsList.append(OrderedDict([
            ("name", arg.name),
            ("type", arg.type_name),
        ]))
        if arg.position:
            argsList[-1]["position"] = arg.position
    out['aruments'] = argsList
    return out

# make yaml happy stuff
from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)

Dumper.add_representer(str,
                       SafeRepresenter.represent_str)

if __name__ == "__main__":
    to_dump = [("ls", lsDesc, lsdata), ("pwd", pwdDesc, pwddata), ("cd", cdDesc, cddata),
            ("echo", echoDesc, echodata), ("rm", rmDesc, rmdata), ("mkdir", mkdirDesc, mkdirData),
            ("touch", touchDesc, touchData), ("cat", catDesc, catData)]
    prefix = "../ai-nix-programs/"
    postfix = ".progdesc.hjson"
    for name, desc, data in to_dump:
        descdump = dump_desc(desc)
        descdump['examples'] = get_example_list(data)
        with open(prefix + name + postfix, "w") as f:
            f.write(hjson.dumps(descdump))
            #f.write(yaml.dump(descdump, Dumper=Dumper, default_flow_style=False))
