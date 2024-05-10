


#include <iostream>
#include <utility>
#include "rapidcsv.h"
#include "data_loader.cpp"


class LinearRegression {


public:
    float *weights;
    float bias;

    explicit LinearRegression(std::string filename) {
        data = CSVDataLoader(std::move(filename)).load_csv();
        weights = new float[data.GetColumnCount()];
        bias = 0.0f;
    }

    ~LinearRegression() {
        delete[] weights;
    }


private:
    rapidcsv::Document data;

};