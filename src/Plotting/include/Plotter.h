#pragma once

#ifndef PLOTTER_H
#define PLOTTER_H

#include "matplotlibcpp.h"
#include <iostream>
#include <fstream>

namespace plt = matplotlibcpp;

class Plotter
{
    private:

        std::vector<float> dataV;
        std::vector<float> dataV2;

    public:
        Plotter();
        
        template <typename T>
        void getSurface(std::vector<T>& vec1, std::vector<T>& vec2, std::vector<T>& vec3, const char* fileName1 = "data.txt", const char* fileName2 = "data2.txt", const char* fileName3 = "data3.txt")
        {
            getFile<T>(vec1, fileName1);
            getFile<T>(vec2, fileName2);
            getFile<T>(vec3, fileName3);
            plotSurface<T>(vec1, vec2, vec3);
        }

        template <typename T>
        void getFile(std::vector<T>& vec, const char* fileName = "data.txt")
        {
            std::string file_path = __FILE__;
            std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
            std::string data_path = dir_path + "/../data/" + fileName;

            getData<T>(data_path, vec);
        }
        
        template <typename T>
        void getData(std::string& filePath, std::vector<T>& vec)
        {
            std::fstream datafile(filePath, std::ios_base::in);

            if (!datafile)
                std::cout << "File not Found in path : " << filePath << '\n';

            T a;
            while ( datafile >> a)
            {
                vec.push_back(a);
            }
        }


        void plotGraph();
        void plotVec();
        template <typename T>
        void plotSurface(std::vector<T>& vec1, std::vector<T>& vec2, std::vector<T>& vec3)
        {
            plt::plot(vec1,vec2);
            plt::legend();
            plt::show();
        }
        
        

};






#endif