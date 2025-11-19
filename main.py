from Matrix_Multiplication import standard_multiplication, strassen_multiply
import time


def main():
    a = [[2, 2], [6, 1]]
    b = [[3, 2], [4, 3]]

    start_time_standard = time.perf_counter()
    c = standard_multiplication(a, b)
    end_time_standard = time.perf_counter()
    duration_standard = end_time_standard - start_time_standard

    start_time_strassen = time.perf_counter()
    d = strassen_multiply(a, b)
    end_time_strassen = time.perf_counter()
    duration_strassen = end_time_strassen - start_time_strassen

    print("Standard multiplication results:\n")
    for row in c:
        print(row)
    print("Time: " + str(duration_standard))

    print("\n\nstrassen results:\n")
    for row in d:
        print(row)
    print("Time: " + str(duration_strassen))

if __name__ == '__main__':
    main()
