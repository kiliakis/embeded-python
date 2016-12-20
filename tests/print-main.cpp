#include <lib-python.h>
#include <iostream>
#include <vector>
int main(int argc, char *argv[])
{
    // Py_Initialize();
    // import_array1(-1);
    python::initialize();

    python::print("Hello World!");
    python::print(argv[0]);

    // int a[4] = {1, 2, 3, 4};
    // std::cout << "The result is... " <<
    //           python::accumulate(a, 4) << "\n";

    // double b[4] = {1.7, 2.32, 4.5, 0.0};

    // std::cout << "The result is... " <<
    //           python::accumulate(b, 4) << "\n";

    // double c[] = {1.7, 2.32, 4.5, 0.0, -1, 10.0, 2, 2};

    // auto index = python::where_bigger_than(c, 8, 2.0);
    // for (const auto &i : index)
    //    std::cout << i << "\n";

    // index = python::where_less_than(c, 8, 2.0);
    // for (const auto &i : index)
    //    std::cout << i << "\n";

    std::map<std::string, python::multi_t> map;
    map["number"] = {42.0};
    map["int"] = {42};
    map["list"] = {std::vector<double> {1.0, 2.0, 3, 4, 5, 15}};
    pass_dictionary(map);
    return 0;
}
