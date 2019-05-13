import numpy as np
import task_info as TI



DELIMETER = ","



def generate_logical(rows_count, function, filename="logical.csv"):
    random_data = np.random.rand(rows_count, 3)
    random_logical = []

    for row in random_data:
        random_logical.append((list(map(lambda x: int(round(x)), row))))
    
    with open(filename, "w") as file:
        file.write(DELIMETER.join(["x", "y", "z", "result"]) + "\n")
        for row in random_logical:
            row.append(function(*row))
            file.write(DELIMETER.join(map(str, row)) + "\n")



if __name__ == "__main__":
    generate_logical(100, TI.second_function)
