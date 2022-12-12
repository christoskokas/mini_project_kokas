#include "LocalBA.h"

namespace vio_slam
{

LocalMapper::LocalMapper(Map* _map) : map(_map)
{

}

void LocalMapper::beginLocalMapping()
{
    while ( !map->endOfFrames )
    {
        
    }
}



} // namespace vio_slam