#include <iostream>
#include "include/market_data.cuh"

int main() {
    try {
        TickDataSoA data(0);
        std::cout << "TickDataSoA with size 0 created successfully" << std::endl;
        std::cout << "size: " << data.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}