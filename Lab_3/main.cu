
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
        // 1 1 1 1 2 5 5
        //   1 1 1 1 2 5
        // Hint: inner_product, plus<int>, not_equal_to<int>
        //
        // wynik = start Op1 (A_0 Op2 B_0) Op1 (A_1 Op2 B_1)
        // Przesunięcie o +1 sprawia, że porównujemy każdy element z jego lewym sąsiadem.
        // Op2 (not_equal_to) zwraca 1 przy wykryciu zmiany dnia, a Op1 (plus) dodaje te jedynki do sumy.
        // Wartość start = 1 uwzględnia pierwszy dzień, który nie ma poprzednika.

        int result = thrust::inner_product(
            data.day.begin() + 1,
            data.day.end(),
            data.day.begin(),
            1,                           // start (wartość początkowa akumulatora)
            thrust::plus<int>(),         // Op1: agreguje wyniki cząstkowe (dodawanie jedynek)
            thrust::not_equal_to<int>()  // Op2: funkcja na parach (1 gdy A_i != B_i, 0 gdy równe)
        );
        int how_much_not_rainy_days = data.day[data.day.size()-1]+1 - result;
        cout<<"Zadanie 1 \n"<<how_much_not_rainy_days<<endl;
    }

    //  Calculate total rainfall at each site
    {
        // Hint: sort_by_key, inner_product, plus<int>, not_equal_to<int>, reduce_by_key
        // ToDo
        // kopie wektorow
        thrust::device_vector<int> measure_copy = data.measurement;
        thrust::device_vector<int> sites_copy = data.site;
        // sortujemy oba wektory wg mmiejscaa, zamieniamy analogicznie i w site i odpowiadajace element w measure
        thrust::sort_by_key(sites_copy.begin(), sites_copy.end(), measure_copy.begin());
        // liczymy unikalne sites ( sites jest posortowane wiec identycznie jak w zadaniu 1)
        int unique_sites = thrust::inner_product(
            sites_copy.begin() + 1,
            sites_copy.end(),
            sites_copy.begin(),
            1,
            thrust::plus<int>(),
            thrust::not_equal_to<int>()
            );
        // tworzymy wektore wynikowe
        thrust::device_vector<int> keys(unique_sites, 0);
        thrust::device_vector<int> values(unique_sites, 0);
        // rob
        thrust::reduce_by_key(
            sites_copy.begin(),
            sites_copy.end(),
            measure_copy.begin(),
            keys.begin(),
            values.begin(),
            thrust::equal_to<int>()
            );
        cout<<"Zadanie 2 \n";
        for (int i=0;i<unique_sites;i++) {
            cout<<keys[i]<<" "<<values[i]<<endl;
        }
    }

//  Get number of days when rainfall exceeded 5
    {
    // ToDo
    // Hint: reduce_by_key, count_if, lambda function
    cout<<"Zadanie 3 \n";
        thrust::device_vector<int> unique_days(data.day.size());
        thrust::device_vector<int> daily_rainfall(data.day.size());
        auto new_end = thrust::reduce_by_key(
        data.day.begin(),
        data.day.end(),
        data.measurement.begin(),
        unique_days.begin(),
        daily_rainfall.begin()
    );
        int count = thrust::count_if(
        daily_rainfall.begin(),
        daily_rainfall.end(),
        }
    );


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

