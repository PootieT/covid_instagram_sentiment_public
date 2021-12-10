def parseLine(line):
    data = line.split()
    score = float(data[-1])
    sentence = ' '.join(data[1:-2])

    return sentence, score


def readData(pth):

    with open(pth, 'r') as file:
        data = file.read()
        data = data.strip('\n')

    data = [parseLine(line) for line in data.split('\n')]
    return data
