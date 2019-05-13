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



if __name__ == "__main__":
    generate_logical(100, TI.second_function)
