import data as sampledata
from program_description import Argument, AIProgramDescription
from cmd_parse import CmdParser, CmdParseError
from bashmetrics import BashMetric
import argparse
def nonascii_untokenize(s):
    s = s.replace(" ", "")
    s = s.replace("<SPACE>", " ")
    return s


parsefaildesc = AIProgramDescription(
    name = "astparsefail", arguments = []
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('eval')
    parser.add_argument('ground_truth')
    args = parser.parse_args()
    print(args)
    with open(args.eval) as file:
        eval_parses = [nonascii_untokenize(l.strip()) for l in file]
    with open(args.ground_truth) as file:
        gt_parses = [nonascii_untokenize(l.strip()) for l in file]

    parser = CmdParser(sampledata.all_descs + [parsefaildesc])
    metric = BashMetric()
    metric.reset()
    parseFails = 0
    
    for pred, gt in zip(eval_parses, gt_parses):
        print("pred:",pred)
        print("gt:", gt)
        try:
            predAst = parser.parse(pred)
        except:
            parseFails += 1
            predAst = parser.parse("astparsefail")
        gtAst = parser.parse(gt)
            
        metric.update(([predAst], [gtAst]))
    print("""Validation Results:
        FirstCmdAcc: {:.2f} ArgAcc: {:.2f} ValExactAcc: {:.2f} ExactAcc: {:.2f}"""
        .format(metric.first_cmd_acc(), metric.arg_acc(),
            metric.arg_val_exact_acc(), metric.exact_match_acc()))
    print("Produced invalid commands", parseFails)
