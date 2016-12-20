#ifndef LIB_PYTHON_H_
#define LIB_PYTHON_H_
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/core/include/numpy/arrayobject.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace python {
    struct multi_t {
        double d;
        std::string s;
        int i;
        std::vector<double> v;
        std::string type;
        multi_t() {}
        multi_t(double _d) : d(_d), type("double") {}
        multi_t(std::string _s) : s(_s), type("string") {}
        multi_t(int _i) : i(_i), type("int") {}
        multi_t(const std::vector<double> &_v) : v(_v), type("f_vector_t") {}
    };

    void print(const std::string &text);
    int accumulate(int *array, const int length);
    double accumulate(double *array, const int length);
    int initialize();
    void finalize();
    std::vector<int> where_bigger_than(double *array, const int length, double val);
    std::vector<int> where_less_than(double *array, const int length, double val);
    void add_one(int *array, const int length);
    void simple_plot();
    PyObject *convert_double(double value);
    PyObject *convert_int(int value);
    PyObject *convert_string(std::string &value);
    PyObject *convert_dictionary(std::map<std::string, multi_t> &map);
    PyArrayObject *convert_double_array(double *array, int size);
    PyObject *convert_double_list(double *array, int size);
    void pass_dictionary(std::map<std::string, multi_t> &map);

}
#endif /* LIB_PYTHON_H_ */
