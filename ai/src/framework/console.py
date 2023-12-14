from framework.keywords import keyword_list


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def bold(text: str):
    print(Color.BOLD + text + Color.END)


def highlight(text: str):
    output = []
    lines = text.split('\n')

    for line in lines:
        if line == '':
            continue

        is_found = False
        for kw in keyword_list:
            word = kw + ':'
            index = line.find(word)
            if index != -1:
                is_found = True
                output.append(line[:index] + Color.BOLD + word + Color.END +
                              Color.GREEN + line[index + len(word):] + Color.END)
        if not is_found:
            output.append(Color.GREEN + line + Color.END)

    print('\n'.join(output))
