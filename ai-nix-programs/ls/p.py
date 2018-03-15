from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open('program.yaml', 'r') as f:
    data = load(f, Loader=Loader)
print(data)
