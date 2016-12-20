#include <lib-python.h>
#include <iostream>


int main(int argc, char *argv[])
{
   python::initialize();

   python::print("Hello World!");
   python::print(argv[0]);

   int *a = new int[4];
   for (int i = 0; i < 4; ++i)
      a[i] = i;

   python::add_one(a, 4);

   std::cout << "The result is... \n";
   for (int i = 0; i < 4; ++i)
      std::cout << a[i] << "\n";

   return 0;
}