import sys
import re


def tokenise_file(filename):
    with open(filename, "r") as f:
        with open(filename + ".tkn", "w") as out_file:
            for line in f:
                tokenised = tokenise(line)
                out_file.write(tokenised + "\n")
    print("written to '{}'".format(filename + ".tkn"))


def tokenise(line):
    tokens = re.findall(r"<s>|[\w]+|[.,!?;'\"]|</s>", line)
    return " ".join(tokens)


if __name__ == "__main__":
    tokenise(input())
