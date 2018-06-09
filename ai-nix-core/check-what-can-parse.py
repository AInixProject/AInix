from __future__ import print_function
import data as sampledata
from program_description import Argument, AIProgramDescription
from cmd_parse import CmdParser, CmdParseError
from bashmetrics import BashMetric
import argparse
from colorama import init
init()
from colorama import Fore, Back, Style
parsefaildesc = AIProgramDescription(
    name = "astparsefail", arguments = []
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nl')
    parser.add_argument('cmd')
    args = parser.parse_args()
    print(args)
    with open(args.nl) as file:
        nl_data = [l.strip() for l in file]
    with open(args.cmd) as file:
        cmd_data = [l.strip() for l in file]
    parser = CmdParser(sampledata.all_descs + [parsefaildesc])

    metric = BashMetric()
    metric.reset()
    parseFails = 0
    
    print("lenghts", len(nl_data), len(cmd_data))
    for nl, cmd in zip(nl_data, cmd_data):
        if "`" in cmd or "{" in cmd or "$" in cmd or "#" in cmd:
            parseFails += 1
            continue
        try:
            predAst = parser.parse(cmd)
            print("nl:",nl)
            print("cmd:", cmd)
        except:
            print(Fore.RED, end='')
            print("cmd:", cmd)
            print(Style.RESET_ALL, end='')
            parseFails += 1
    print("parse failse", parseFails, " parse not fail", len(cmd_data) - parseFails)
