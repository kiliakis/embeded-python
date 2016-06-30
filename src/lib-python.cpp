#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <site-packages/numpy/core/include/numpy/arrayobject.h>
#include <lib-python.h>
#include <iostream>


int python::initialize()
{
   Py_Initialize();
   import_array1(0);
   return 0;
}

void python::finalize()
{
   Py_Finalize();
}

void python::print(const std::string &text)
{
   // Build the name object
   auto pModule = PyImport_ImportModule("libpython");
   assert(pModule);

   // // pDict is a borrowed reference
   auto pDict = PyModule_GetDict(pModule);
   assert(pDict);

   // // pFunc is also a borrowed reference
   auto pFunc = PyDict_GetItemString(pDict, "pyprint");
   assert(pFunc);

   if (PyCallable_Check(pFunc)) {
      auto ret = PyObject_CallFunction(pFunc, "s", text.c_str());
      assert(ret);
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
   // std::cout << "we are in the correct function\n";

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


std::vector<int> python::where_bigger_than(double *array, const int length, double val)
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
   result = new long int[len];
   result = reinterpret_cast<long int *>(PyArray_DATA(np_array));

   // Clean up
   Py_DECREF(pModule);

   return std::vector<int>(&result[0], &result[len]);
}


std::vector<int> python::where_less_than(double *array, const int length, double val)
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
   result = new long int[len];
   result = reinterpret_cast<long int *>(PyArray_DATA(np_array));

   // Clean up
   Py_DECREF(pModule);

   return std::vector<int>(&result[0], &result[len]);
}