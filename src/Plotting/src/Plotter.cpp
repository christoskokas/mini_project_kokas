#include "Plotter.h"

Plotter::Plotter()
{

}

// void Plotter::getFile(const char* fileName /* "data.txt" */)
// {
//     std::string file_path = __FILE__;
//     std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
//     std::string data_path = dir_path + "/../data/" + fileName;
//     std::fstream datafile(data_path, std::ios_base::in);
//     if (!datafile)
//         std::cout << "File not Found in path : " << data_path << '\n';
//     float a;
//     while ( datafile >> a)
//     {
//         std::cout << a << "gf ";
//         dataV.push_back(a);
//         dataV2.push_back(a);
//     }
// }

void Plotter::plotGraph()
{
    plt::plot({1,3,2,4});
    plt::show();
}

void Plotter::plotVec()
{
    plt::plot(dataV, dataV2,{{"label", "one"}, {"color", "red"}, {"marker", "o"}, {"linestyle", "--"}});
    plt::plot(dataV2,{{"label", "two"}, {"color", "blue"}, {"marker", "o"}, {"linestyle", "--"}});
    plt::xlabel("xla");
    plt::ylabel("yla");
    plt::legend();
    plt::title("First Trials");
    plt::figure();
    plt::plot(dataV2,{{"label", "two"}, {"color", "blue"}, {"marker", "o"}, {"linestyle", "--"}});
    plt::show();
}

