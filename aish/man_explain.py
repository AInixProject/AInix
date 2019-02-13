import requests
import colorama

WRAP_LEN = 80
BASE_URL = "https://www.mankier.com/api/v2/explain/?cols={WRAP_LEN}&q="


def get_text_from_mankier(query: str):
    resp = requests.get(BASE_URL + query)
    if resp.ok:
        return resp.text
    else:
        return None


def print_man_page_explan(query: str, ref_text: str):
    mankier = get_text_from_mankier
    if mankier is None:
        print("Was not able to get man page explaination")
        return
    print("man page explanation:")
    mak_text = get_text_from_mankier(query)
    highlight_stuff = "-" in ref_text
    for line in mak_text.split("\n")[1:]:
        striped = line.strip()
        if highlight_stuff and striped and striped[0] == "-" and len(striped) < WRAP_LEN:
            # hackily highlight args not in the query
            arg = striped.split()[0]
            if arg not in ref_text:
                print(colorama.Fore.BLUE, end="")
        print(line)
        print(colorama.Fore.RESET, end="")


if __name__ == "__main__":
    print_man_page_explan('ls -l -h | find -name "foo"', "do the show me files")
