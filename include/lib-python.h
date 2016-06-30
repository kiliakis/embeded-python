#ifndef LIB_PYTHON_H_
#define LIB_PYTHON_H_

#include <string>
#include <vector>

namespace python {
   void print(const std::string &text);
   int accumulate(int *array, const int length);
   double accumulate(double *array, const int length);
   int initialize();
   void finalize();
   std::vector<int> where_bigger_than(double *array, const int length, double val);
   std::vector<int> where_less_than(double *array, const int length, double val);
}
#endif /* LIB_PYTHON_H_ */
