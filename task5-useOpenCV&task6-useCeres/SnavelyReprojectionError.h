#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y, double _fx, double _fy, double _cx, double _cy) : observed_x(observation_x),
                                                                           observed_y(observation_y), fx(_fx), fy(_fy), cx(_cx), cy(_cy) {}

    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjection(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    inline bool CamProjection(const T *camera, const T *point, T *predictions) const {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        predictions[0] = T(fx)*p[0]/p[2] + T(cx);
        predictions[0] = T(fy)*p[1]/p[2] + T(cy);
        return true;
    }

    static ceres::CostFunction *Create(double observed_x,double observed_y, double fx, double fy, double cx, double cy) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
            new SnavelyReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }

private:
    double observed_x;
    double observed_y;
    double fx;
    double fy;
    double cx;
    double cy;
};

#endif // SnavelyReprojection.h

