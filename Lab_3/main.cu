
// Using Thrust library.
//
// Assume having data:
//
//    day         [  0  0  1  2  5  5  6  6  7  8 ...]
//    site        [  2  3  0  1  1  2  0  1  2  1 ...]
//    measurement [  9  5  6  3  3  8  2  6  5 10 ...]
//
// For 'day' we have 'measurement' of rainfall in given 'site'.
//
//    Time series sorted by day
//    Measurements of zero are excluded from the time series
//
// Tasks:
//
//    Design structures containing the data (remember the difference AoS and SoA)
//    Calculate number of days with any rainfall
//    Calculate total rainfall at each site
//    Get number of days when rainfall exceeded 5
//    Find the first day when total rainfall exceeded 32
//    Return total rainfall between day 2 and 6 (inclusive)
//
// Note:
//     All computation should be done without loops!
//     Loops are allowed only when printing the results in point "Calculate total rainfall at each site"
//
// Documentation:   https://nvidia.github.io/cccl/unstable/thrust/index.html

#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/binary_search.h>

#include <iostream>
#include <iomanip>

using namespace std;

// Design structures containing the data (remember the difference AoS and SoA)
struct Data
    {
    thrust::device_vector<int> day;
    thrust::device_vector<int> site;
    thrust::device_vector<int> measurement;
    };

int main()
    {
    int day[]     = {  0,  0,  1,  2,  5,  5,  6,  6,  7,  8 };
    int site[]    = {  2,  3,  0,  1,  1,  2,  0,  1,  2,  1 };
    int measure[] = {  9,  5,  6,  3,  3,  8,  2,  6,  5, 10 };

    cout << endl << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl ;

    int SIZE = sizeof(day)/sizeof(int);
    Data data;
    data.day         = thrust::device_vector<int>(day, day+SIZE);
    data.site        = thrust::device_vector<int>(site, site+SIZE);
    data.measurement = thrust::device_vector<int>(measure, measure+SIZE);

//  Calculate number of days with any rainfall
    {
    // ToDo
    // Hint: inner_product, plus<int>, not_equal_to<int>
    }

//  Calculate total rainfall at each site
    {
    // ToDo
    // Hint: sort_by_key, inner_product, plus<int>, not_equal_to<int>, reduce_by_key
    }

//  Get number of days when rainfall exceeded 5
    {
    // ToDo
    // Hint: reduce_by_key, count_if, lambda function
    }

//  Find the first day when total rainfall exceeded 32
    {
    // ToDo
    // Hint: inclusive_scan, lower_bound
    }

//  Return total rainfall between day 2 and 6 (inclusive)
    {
    // ToDo
    // Hint: lower_bound, upper_bound, reduce
    }

    return 0;
    }

