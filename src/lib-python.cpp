#include <lib-python.h>
#include <iostream>
#include <map>
using namespace std;
using namespace python;

int python::initialize()
{
    putenv("PYTHONPATH=" PYTHONPATH);
    Py_SetPythonHome(PYLIBS);
    Py_Initialize();
    import_array1(0);
    return 0;
}

void python::finalize()
{
    Py_Finalize();
}


void python::print(const string &text)
{
    // Py_Initialize();

    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    auto pFunc = PyObject_GetAttrString(pModule, "pyprint");
    assert(pFunc);

    auto pString = PyString_FromString(text.c_str());
    assert(pString);

    if (PyCallable_Check(pFunc)) {
        auto ret = PyObject_CallFunctionObjArgs(pFunc, pString, NULL);
        assert(ret);
    } else {
        PyErr_Print();
    }

    // Clean up
    // Py_DECREF(pModule);
    // Py_Finalize();
}

PyObject *python::convert_double(double value)
{
    auto pVar = PyFloat_FromDouble(value);
    assert(pVar);
    return pVar;
}


PyObject *python::convert_int(int value)
{
    auto pVar = PyInt_FromLong(value);
    assert(pVar);
    return pVar;
}

PyObject *python::convert_string(std::string &value)
{
    auto pVar = PyString_FromString(value.c_str());
    assert(pVar);
    return pVar;
}

PyArrayObject *python::convert_double_array(double *array, int size)
{
    int dims[1] = {size};
    auto pVar = (PyArrayObject *) PyArray_FromDimsAndData(1, dims,
                NPY_DOUBLE, (char *)array);

    assert(pVar);
    return pVar;
}

PyObject *python::convert_double_list(double *array, int size)
{
    auto pList = PyList_New(size);
    assert(pList);
    for (int i = 0; i < size; i++) {
        auto pVar = convert_double(array[i]);
        PyList_SET_ITEM(pList, i, pVar);
    }
    return pList;
}


PyObject *python::convert_dictionary(map<string, multi_t> &map)
{
    if (map.size() > 0) {
        auto pDict = PyDict_New();
        assert(pDict);
        for (auto &pair : map) {
            auto pKey = PyString_FromString(pair.first.c_str());
            if (pair.second.type == "double") {
                auto pVal = convert_double(pair.second.d);
                PyDict_SetItem(pDict, pKey, pVal);
            } else if (pair.second.type == "int") {
                auto pVal = convert_int(pair.second.i);
                PyDict_SetItem(pDict, pKey, pVal);
            } else if (pair.second.type == "string") {
                auto pVal = convert_string(pair.second.s);
                PyDict_SetItem(pDict, pKey, pVal);
            } else if (pair.second.type == "f_vector_t") {
                auto pVal = convert_double_list(pair.second.v.data(),
                                                pair.second.v.size());
                PyDict_SetItem(pDict, pKey, pVal);
            } else {
                cerr << "Warning: type " << pair.second.type
                     << " was not recognized.\n";
            }
        }
        return pDict;
    } else {
        return Py_None;
    }
}


void python::pass_dictionary(map<string, multi_t> &map)
{
    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "print_dict");
    assert(pFunc);
    auto pDict = convert_dictionary(map);

    if (PyCallable_Check(pFunc)) {
        // auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        auto ret = PyObject_CallFunctionObjArgs(pFunc, pDict, NULL);
        assert(ret);
        // auto p = (int *) PyArray_DATA(pArray);
    } else {
        PyErr_Print();
    }
}



void python::add_one(int *array, const int length)
{
    // Both approaches are working, the first is using PyArrayObject,
    // and the second refular PyObjects
    // auto pArray = PyList_New(length);
    // for (int i = 0; i < length; i++) {
    //     auto pVar = PyInt_FromLong(array[i]);
    //     assert(pVar);
    //     PyList_SET_ITEM(pArray, i, pVar);
    // }
    int dims[1] = {length};
    auto pArray = (PyArrayObject *) PyArray_FromDimsAndData(1, dims, NPY_INT,
                  reinterpret_cast<char *>(array));

    // npy_intp dims[1] = {length};
    // auto pArray = PyArray_SimpleNewFromData(1, dims, NPY_INT,
    //                                         reinterpret_cast<void *>(array));

    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "add_one");
    assert(pFunc);

    if (PyCallable_Check(pFunc)) {
        // auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        assert(ret);
        // auto p = (int *) PyArray_DATA(pArray);
    } else {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);

}

int python::accumulate(int *array, const int length)
{

    npy_intp dims[1] = {length};
    auto pArray = PyArray_SimpleNewFromData(1, dims,
                                            NPY_INT, reinterpret_cast<void *>(array));

    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "pyaccumulate");
    assert(pFunc);

    int sum = 0;
    if (PyCallable_Check(pFunc)) {
        auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        assert(ret);
        sum = PyInt_AsLong(ret);
    } else {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);

    return sum;

}


double python::accumulate(double *array, const int length)
{
    npy_intp dims[1] = {length};
    //int dims[1] = {length};
    auto pArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                            (void *)(array));
    assert(pArray);

    // Build the name object
    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);
    // cout << "we are in the correct function\n";

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "pyaccumulate");
    assert(pFunc);

    double sum = 0.0;
    if (PyCallable_Check(pFunc)) {
        // auto pArgs = PyTuple_New(1);
        // PyTuple_SetItem(pArgs, 0, pArray);
        auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        assert(ret);
        sum = PyFloat_AsDouble(ret);
    } else {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);

    return sum;
}


vector<int> python::where_bigger_than(double *array, const int length, double val)
{

    long int *result = NULL;

    // Load the module object
    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "where_bigger_than");
    assert(pFunc);

    npy_intp dims[1] = {length};
    auto pArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                            (void *)(array));
    assert(pArray);

    auto pVal = PyFloat_FromDouble(val);

    if (not PyCallable_Check(pFunc)) {
        PyErr_Print();
        exit(-1);
    }

    auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, pVal, NULL);
    assert(ret);

    auto np_array = (PyArrayObject *)(ret);
    int len = PyArray_SHAPE(np_array)[0];
    // result = new long int[len];
    result = reinterpret_cast<long int *>(PyArray_DATA(np_array));

    // Clean up
    Py_DECREF(pModule);

    return vector<int>(&result[0], &result[len]);
}


vector<int> python::where_less_than(double *array, const int length, double val)
{

    long int *result = NULL;

    // Load the module object
    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);

    // pFunc is also a borrowed reference
    auto pFunc = PyObject_GetAttrString(pModule, "where_less_than");
    assert(pFunc);

    npy_intp dims[1] = {length};
    auto pArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                            (void *)(array));
    assert(pArray);

    auto pVal = PyFloat_FromDouble(val);

    if (not PyCallable_Check(pFunc)) {
        PyErr_Print();
        exit(-1);
    }

    auto ret = PyObject_CallFunctionObjArgs(pFunc, pArray, pVal, NULL);
    assert(ret);

    auto np_array = (PyArrayObject *)(ret);
    int len = PyArray_SHAPE(np_array)[0];
    // result = new long int[len];
    result = reinterpret_cast<long int *>(PyArray_DATA(np_array));

    // Clean up
    Py_DECREF(pModule);

    return vector<int>(&result[0], &result[len]);
}


void python::simple_plot()
{
    auto pModule = PyImport_ImportModule("libpython");
    assert(pModule);
    auto pFunc = PyObject_GetAttrString(pModule, "simple_plot");
    assert(pFunc);
    if (not PyCallable_Check(pFunc)) {
        PyErr_Print();
        exit(-1);
    }
    auto ret = PyObject_CallFunctionObjArgs(pFunc, NULL);
    assert(ret);
    Py_DECREF(pModule);

}