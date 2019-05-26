import numpy as np
import task_info as TI



DELIMETER = ","



def generate_logical(rows_count, function, filename="logical.csv"):
    random_logical = np.random.randint(0, 2, (rows_count, 3))
    
    with open(filename, "w") as file:
        file.write(DELIMETER.join(["x", "y", "z", "result"]) + "\n")
        for row in random_logical:
            row = np.append(row, function(*row))
            file.write(DELIMETER.join(map(str, row)) + "\n")


def generate_regression(rows_count, function, min=-10, max=10, filename="regression.csv"):
    x_random = np.random.rand(rows_count);
    x_in_interval = map(lambda x: (max - min) * x + min, x_random)

    with open(filename, "w") as file:
        file.write(DELIMETER.join(["x", "y"]) + "\n")
        for x in x_in_interval:
            row = (x, function(x))
            file.write(DELIMETER.join(map(str, row)) + "\n")


def read_csv(filename, data_type):
    results = []
    with open(filename, "r") as file:
        file.readline()
        for line in file.readlines():
            row = line.rstrip().split(DELIMETER)
            results.append(list(map(data_type, row)))
    return results



if __name__ == "__main__":
    generate_regression(100, TI.first_function)
    generate_logical(100, TI.second_function)
