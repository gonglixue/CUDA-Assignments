#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>

template<typename T>
void swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}



#endif