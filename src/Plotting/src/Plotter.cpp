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
    // std::vector<std::vector<double>>v1,v2,v3;
    // std::vector<double>x,y,z;
    // x.push_back(1);
    // x.push_back(2);
    // x.push_back(3);
    // y.push_back(1);
    // y.push_back(2);
    // y.push_back(3);
    // z.push_back(1);
    // z.push_back(2);
    // z.push_back(3);
    std::vector<double> x, y, z;
    double theta, r;
    double z_inc = 4.0/99.0; double theta_inc = (8.0 * M_PI)/99.0;
    
    for (double i = 0; i < 100; i += 1) {
        theta = -4.0 * M_PI + theta_inc*i;
        z.push_back(-2.0 + z_inc*i);
        r = z[i]*z[i] + 1;
        x.push_back(r * sin(theta));
        y.push_back(r * cos(theta));
    }
    // plt::plot_surface(v1,v2,v3);
    std::map<std::string, std::string> keywords;
    keywords.insert(std::pair<std::string, std::string>("label", "parametric curve") );
    plt::plot3(x,y,z, keywords);
    plt::xlabel("x label");
    plt::ylabel("y label");
    plt::set_zlabel("z label"); // set_zlabel rather than just zlabel, in accordance with the Axes3D method
    plt::legend();
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

