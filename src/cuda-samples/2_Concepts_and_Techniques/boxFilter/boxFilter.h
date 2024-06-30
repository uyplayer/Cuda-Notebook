//
// Created by uyplayer on 2024-06-30.
//

#ifndef BOXFILTER_H
#define BOXFILTER_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat boxFilterCpu(const cv::Mat &input, int kernelSize) {
    std::cout << "Running box filter on CPU" << std::endl;
    if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }
    int kernelRadius = kernelSize / 2;
    cv::Mat output = input.clone();

    for (int row = 0; row < input.rows; ++row) {
        for (int col = 0; col < input.cols; ++col) {
            cv::Vec3d sum(0, 0, 0);
            int count = 0;

            for (int x = -kernelRadius; x <= kernelRadius; ++x) {
                for (int y = -kernelRadius; y <= kernelRadius; ++y) {
                    int newRow = row + x;
                    int newCol = col + y;

                    if (newRow >= 0 && newRow < input.rows && newCol >= 0 && newCol < input.cols) {
                        if (input.channels() == 1) {
                            sum[0] += input.at<uchar>(newRow, newCol);
                        } else if (input.channels() == 3) {
                            sum += input.at<cv::Vec3b>(newRow, newCol);
                        }
                        count++;
                    }
                }
            }

            if (input.channels() == 1) {
                output.at<uchar>(row, col) = static_cast<uchar>(sum[0] / count);
            } else if (input.channels() == 3) {
                output.at<cv::Vec3b>(row, col) = sum / count;
            }
        }
    }

    return output;
}


#endif //BOXFILTER_H
