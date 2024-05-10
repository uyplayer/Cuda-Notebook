
#include <iostream>
#include <utility>
#include "rapidcsv.h"


class CSVDataLoader {
public:
    explicit CSVDataLoader(std::string filename) : filename(std::move(filename)) {
        if (this->filename.find(".csv") == std::string::npos) {
            std::cerr << "File is not in CSV format" << std::endl;
        }
    }

    ~CSVDataLoader() = default;

    rapidcsv::Document load_csv() {
        rapidcsv::Document doc(filename, rapidcsv::LabelParams(),
                               rapidcsv::SeparatorParams(),
                               rapidcsv::ConverterParams(true));
    }


private:
    std::string filename;

};