def read_csv(filepath: str) -> list:
    output = []
    with open(filepath) as f:
        for line in f:
            data = line.replace("\n", "")
            output.append(data.split(','))
    return output


def write_csv(filepath: str, data: list):
    with open(filepath, "w") as f:
        for row in data:
            line = ''
            for value in row:
                if len(line) > 0:
                    line = line + ','
                line = line + str(value)
            f.write(line + '\n')
