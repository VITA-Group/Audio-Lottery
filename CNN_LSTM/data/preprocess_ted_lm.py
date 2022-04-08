import sys
import nltk
from pathlib import Path

if __name__ == '__main__':
    ted_root = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    txt_dir = ted_root.joinpath('TEDLIUM_release2', 'train', 'converted', 'txt')
    with open(output_path, 'w') as f_out:
        for txt_path in txt_dir.glob('*.txt'):
            with open(txt_path, 'r') as f_in:
                line = f_in.readline()
            sentence = nltk.sent_tokenize(line)[0]
            f_out.write(' '.join(nltk.word_tokenize(sentence)).upper() + '\n')
