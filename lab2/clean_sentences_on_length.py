import sys


def clean_sentences_on_length(file1, file2, min_len, max_len):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            with open(file1 + ".cln", "w") as f1w:
                with open(file2 + ".cln", "w") as f2w:
                    for line1 in f1:
                        line2 = f2.readline()
                        if valid_sentence(line1, min_len, max_len) and valid_sentence(line2, min_len, max_len):
                            f1w.write(line1)
                            f2w.write(line2)
    print("files '{}' and '{}' written.".format(file1 + ".cln", file2 + ".cln"))

def valid_sentence(line, min_len, max_len):
    tokens = line.strip().split()
    if len(tokens) > int(min_len) and len(tokens) < int(max_len):
        return True

if __name__ == "__main__":
    clean_sentences_on_length(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
