import glob, json, os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

import re

import sys, token, tokenize

def do_file(fname):
    source = open(fname)
    mod = open(fname.replace(".py", "_cleaned.py"), "w")

    prev_toktype = token.INDENT
    first_line = None
    last_lineno = -1
    last_col = 0

    tokgen = tokenize.generate_tokens(source.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if 0:   # Change to if 1 to see the tokens fly by.
            print("%10s %-14s %-20r %r" % (
                tokenize.tok_name.get(toktype, toktype),
                "%d.%d-%d.%d" % (slineno, scol, elineno, ecol),
                ttext, ltext
                ))
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            mod.write(" " * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            # Docstring
            mod.write("")
        elif toktype == tokenize.COMMENT:
            # Comment
            mod.write("")
        else:
            mod.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--segment_len', type=int, default=254,
                        help='the length of each example')
    # we set this to be 254 instead of 256 because we want the input to be like: <control_code> input_ids <eos>
    parser.add_argument('--stride', type=int, default=10,
                        help='stride to split training examples')
    parser.add_argument('--dev_size', type=float, default=0.1,
                        help='split ratio of development set for each language')
    args = parser.parse_args()

    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
    segments = {}

    

    path = '/Users/seannelson/dissertation/algorithms/Python'
    source_files = glob.glob(f'{path}/**/*.py', recursive=True)
    for each_src in tqdm(source_files):
        if "cleaned.py" in each_src:
            with open(each_src, "r", encoding="utf-8") as f:
                code_content = f.read()
                print(code_content)
                encoded = gpt2_tok.encode(code_content)
                for i in range(len(encoded) // args.stride):
                    seg = encoded[i * args.stride:i * args.stride + args.segment_len]
                    if path not in segments:
                        segments[path] = []
                    segments[path].append(json.dumps({"token_ids": seg, "label": path}))

    # originally splits the segements into train and test sets
    temp_train, test, train, dev = [], [], [], []
    for key in segments:
        # we don't shuffle before splitting because we want the train and dev to be very different (less overlapping)
        temp_tr, te = train_test_split(segments[key], test_size=args.dev_size)
        test += te

        # splits the temp_train set into the train and validation sets

        tra, de = train_test_split(temp_tr, test_size=args.dev_size)
        train += tra
        dev += de


    to_path = "source_code/json"
    if not os.path.isdir(to_path):
        os.makedirs(to_path)

    with open(os.path.join(to_path, "train.jsonl"), "w") as f:
        f.write("\n".join(train))

    with open(os.path.join(to_path, "dev.jsonl"), "w") as f:
        f.write("\n".join(dev))

    with open(os.path.join(to_path, "eval.jsonl"), "w") as f:
        f.write("\n".join(test))


def clean_comments_from_files():
    path = '/Users/seannelson/dissertation/algorithms/Python'
    source_files = glob.glob(f'{path}/**/*.py', recursive=True)
    for each_src in tqdm(source_files):
        do_file(each_src)
