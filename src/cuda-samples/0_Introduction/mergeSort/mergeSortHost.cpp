//
// Created by uyplayer on 2024-06-23.
//


#include <iostream>
#include<vector>

void merge(std::vector<int> &nums, int left, int middle, int right) {
    std::vector<int> tmp(right - left + 1);
    int i = left;
    int j = middle + 1;
    int k = 0;
    while (i <= middle && j <= right) {
        if (nums[i] < nums[j]) {
            tmp[k++] = nums[i++];
        } else {
            tmp[k++] = nums[j++];
        }
    }
    while (i <= middle) {
        tmp[k++] = nums[i++];
    }
    while (j <= right) {
        tmp[k++] = nums[j++];
    }
    for (int i = left, k = 0; i <= right; i++, k++) {
        nums[i] = tmp[k];
    }
}

void mergeSort(std::vector<int> &nums, int left, int right) {
    if (left <= right) {
        int middle = left + (right - left) / 2;
        mergeSort(nums, left, middle);
        mergeSort(nums, middle + 1, right);
        merge(nums, left, middle, right);
    }
}
