#pragma once

#ifndef SYSTEM_H
#define SYSTEM_H

#include "Settings.h"
#include "Camera.h"
#include "trial.h"
#include "Frame.h"
#include "Map.h"
#include "LocalBA.h"
#include <thread>
#include <string>



namespace vio_slam
{

class System
{

    private:

    public:

        System(std::string& confFile);

        void SLAM();

        std::thread* Visual;
        std::thread* Tracking;
        std::thread* LocalMapping;

        Frame* mFrame;

        Zed_Camera* mZedCamera;

        ConfigFile* mConf;

        RobustMatcher2* mRb;

        Map* map;

        LocalMapper* localMap;

};

} // namespace vio_slam



#endif // SYSTEM_H