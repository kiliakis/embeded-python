
def pyprint(string):
    # import sys
    # print sys.prefix
    # import numpy
    # print numpy.__file__
    print string


def pyaccumulate(list):
    from numpy import sum
    return sum(list)


def where_bigger_than(list, num):
    from numpy import where, array
    a = where(array(list) > num)[0]
    return a


def where_less_than(list, num):
    from numpy import where, array
    a = where(array(list) < num)[0]
    return a


def add_one(array):
    for i in range(len(array)):
        array[i] += 1
    # print(array.PyCapsule_GetPointer)
    return 1


def simple_plot():
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
    plt.show()


def print_dict(d):
    print(d)
    for k,v in d.items():
        print("key: ", k)
        print("value: ", v)
