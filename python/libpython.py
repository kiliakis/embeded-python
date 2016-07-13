
def pyprint(string):
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


def add_one(a):
    for i in range(len(a)):
        a[i] += 1
    return 1
