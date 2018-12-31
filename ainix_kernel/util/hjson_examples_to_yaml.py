"""Quick and dirty code for converting from a the program descriptions
to the yaml example format (this will surely change though...."""
import sys

import hjson
#import yaml
from ruamel.yaml import YAML

from collections import OrderedDict

if __name__ == "__main__":
    program = "cat"
    with open(f"../../ai-nix-programs/{program}.progdesc.hjson", "rb") as f:
        v = hjson.load(f)
    out_examples = []
    for e in v['examples']:
        out_examples.append({
            "x": e["lang"],
            "y": e["cmd"],
        })
    full_out = {"defines": [dict(
        define_new="example_set",
        y_type="Program",
        examples=out_examples
    )]}
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    with open(f"../../builtin_types/{program}_examples.ainix.yaml", "wb") as f:
        print(yaml.dump(full_out, f))
