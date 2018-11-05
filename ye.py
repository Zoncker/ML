def main():
    rows = int(input("Enter the number of rows: "))
    columns = int(input("Enter the number of columns: "))
    matrix = []
    print("Enter the %s x %s matrix: " % (rows, columns))
    for i in range(rows):
        matrix.append(list(map(int, input().rstrip().split())))
    res = gauss_m(matrix)
    print(res[0])


def gauss_m(m):
    # eliminate columns
    for col in range(len(m[0])):
        for row in range(col+1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    result = []
    m.reverse() # makes it easier to backsolve
    for sln in range(len(m)):
            if sln == 0:
                result.append(m[sln][-1] / m[sln][-2])
            else:
                inner_coefs = 0
                # substitute in all known coefficients
                for x in range(sln):
                    inner_coefs += (result[x]*m[sln][-2-x])
                # the equation is now reduced to ax + b = c form
                # solve with (c - b) / a
                result.append((m[sln][-1]-inner_coefs)/m[sln][-sln-2])
    result.reverse()
    return result


if __name__ == '__main__':
    main()
