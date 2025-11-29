from Matrix_Multiplication import standard_multiplication, strassen_multiply, random_matrix
import time


def main():
    print("\n\nWhat are the sizes of the matrices you wish to multiply? A x B, B x C")
    A = int(input("\nA = "))
    B = int(input("\nB = "))
    C = int(input("\nC = "))

    print("\nDo you want to see the final product? y/n: ")
    choice = input()

    print("\nWhat values do you wish for the matrices to have?")
    print("\n1 = All values will be 1")
    print("\n2 = All values will be 2")
    print("\n3 = All values will be between 1 and 10")
    print("\n4 = All values will be between 1000 and 9999")
    value = int(input("\n1/2/3/4: "))

    matrix1 = random_matrix(A, B, value)
    matrix2 = random_matrix(B, C, value)

    print("\nMatrix 1:")
    for row in matrix1:
        print (row)

    print("\nMatrix 2:")
    for row in matrix2:
        print (row)

    start_time_standard = time.perf_counter()
    c = standard_multiplication(matrix1, matrix2)
    end_time_standard = time.perf_counter()
    duration_standard = end_time_standard - start_time_standard

    start_time_strassen = time.perf_counter()
    d = strassen_multiply(matrix1, matrix2)
    end_time_strassen = time.perf_counter()
    duration_strassen = end_time_strassen - start_time_strassen

    print("Standard multiplication results:\n")
    if choice == "y":
        for row in c:
            print(row)
    print("Time: " + str(duration_standard))

    print("\n\nstrassen results:\n")
    if choice == "y":
        for row in d:
            print(row)
    print("Time: " + str(duration_strassen))

if __name__ == '__main__':
    main()
