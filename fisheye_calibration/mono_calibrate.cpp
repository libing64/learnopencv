#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

vector<vector<Point3d>> object_points;
vector<vector<Point2f>> imagePoints;
vector<Point2f> corners;
vector<vector<Point2d>> img_points;

Mat img, gray, spl;

void load_image_points(int board_width, int board_height, float square_size, int num_imgs,
                       char *img_dir, char *img_filename)
{
    Size board_size = Size(board_width, board_height);
    int board_n = board_width * board_height;

    for (int i = 1; i <= num_imgs; i++)
    {
        char left_img[100];
        sprintf(left_img, "%s%s%d.jpg", img_dir, img_filename, i);
        img = imread(left_img, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        bool found = false;

        found = cv::findChessboardCorners(img, board_size, corners,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if (found)
        {
            cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray, board_size, corners, found);
        }
        vector<cv::Point3d> obj;
        for (int i = 0; i < board_height; ++i)
            for (int j = 0; j < board_width; ++j)
                obj.push_back(Point3d(double((float)j * square_size), double((float)i * square_size), 0));

        if (found)
        {
            cout << i << ". Found corners!" << endl;
            imagePoints.push_back(corners);
            object_points.push_back(obj);
        }
    }
    for (int i = 0; i < imagePoints.size(); i++)
    {
        vector<Point2d> v1;
        for (int j = 0; j < imagePoints[i].size(); j++)
        {
            v1.push_back(Point2d((double)imagePoints[i][j].x, (double)imagePoints[i][j].y));
        }
        img_points.push_back(v1);
    }
}

//! [compute_errors]
static double computeReprojectionErrors( const vector<vector<Point3d> >& objectPoints,
                                         const vector<vector<Point2d> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         Mat & cameraMatrix , Mat & distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<Point2d> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i )
    {

        fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
                                distCoeffs);
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

int main(int argc, char const *argv[])
{
    int board_width, board_height, num_imgs;
    float square_size;
    char *img_dir;
    char *img_filename;
    char *rightimg_filename;
    char *out_file;

    static struct poptOption options[] = {
        {"board_width", 'w', POPT_ARG_INT, &board_width, 0, "Checkerboard width", "NUM"},
        {"board_height", 'h', POPT_ARG_INT, &board_height, 0, "Checkerboard height", "NUM"},
        {"square_size", 's', POPT_ARG_FLOAT, &square_size, 0, "Checkerboard square size", "NUM"},
        {"num_imgs", 'n', POPT_ARG_INT, &num_imgs, 0, "Number of checkerboard images", "NUM"},
        {"img_dir", 'd', POPT_ARG_STRING, &img_dir, 0, "Directory containing images", "STR"},
        {"img_filename", 'l', POPT_ARG_STRING, &img_filename, 0, "Left image prefix", "STR"},
        {"out_file", 'o', POPT_ARG_STRING, &out_file, 0, "Output calibration filename (YML)", "STR"},
        POPT_AUTOHELP{NULL, 0, 0, NULL, 0, NULL, NULL}};

    POpt popt(NULL, argc, argv, options, 0);
    int c;
    while ((c = popt.getNextOpt()) >= 0)
    {
    }

    load_image_points(board_width, board_height, square_size, num_imgs, img_dir, img_filename);

    printf("Starting Calibration\n");
    Mat K;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    Mat D;
    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::fisheye::CALIB_FIX_SKEW;

    Mat _rvecs, _tvecs;
    Size img_size = Size(img.cols, img.rows);

    cout << "object_points: " << object_points.size() << endl;
    cout << "img_points: " << img_points.size() << endl;
    cout << "img_size: " << img_size << endl;
    cout << "flag: " << flag << endl; 

    cv::fisheye::calibrate(object_points, img_points, img_size,
                                 K, D, _rvecs, _tvecs, flag);

    cout << "camera_matrix: " << K << endl;
    cout << "distortion coeffs: " << D << endl;

    rvecs.reserve(_rvecs.rows);
    tvecs.reserve(_tvecs.rows);
    for(int i = 0; i < int(object_points.size()); i++)
    {
        rvecs.push_back(_rvecs.row(i));
        tvecs.push_back(_tvecs.row(i));
    }

    vector<float> reprojErrs;
    float total_avg_err = computeReprojectionErrors(object_points, img_points, rvecs, tvecs, K,
                                            D, reprojErrs);
    cout << "rms: " << total_avg_err << endl;
    cv::FileStorage fs(out_file, cv::FileStorage::WRITE);
    fs << "K" << Mat(K);
    fs << "D" << D;
    fs << "rvecs" << _rvecs;
    fs << "tvecs" << _tvecs;
    fs << "rms" << total_avg_err;
    printf("Done Calibration\n");

    return 0;
}
