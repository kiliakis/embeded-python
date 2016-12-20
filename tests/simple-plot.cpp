#include <lib-python.h>
#include <iostream>


int main(int argc, char *argv[])
{
   python::initialize();

   python::simple_plot();

   return 0;
}