
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <numeric>

using namespace cv;
using namespace std;

bool isDuplicate(int cx, int cy, const vector<tuple<double, int, int>>& seen, int thresh = 10) {
    for (const auto& [area, sx, sy] : seen) {
        if (abs(cx - sx) < thresh && abs(cy - sy) < thresh) {
            return true;
        }
    }
    return false;
}

int main() {
    string video_source = "../sample.mp4";
    VideoCapture cap(video_source);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream." << endl;
        return -1;
    }
    
    // Only considering White color for now
    map<string, tuple<Scalar, Scalar, Scalar>> color_ranges = {
        {"white", {Scalar(0, 0, 175), Scalar(180, 40, 255), Scalar(255, 255, 255)}}
    };

    vector<double> frame_times;
    int frame_index = 0;

    while (cap.isOpened()) {
        auto start = chrono::high_resolution_clock::now();

        Mat frame;
        if (!cap.read(frame)) break;

        Mat hsv, h, s, v;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        vector<Mat> hsv_channels;

        // Split HSV channels, Equalize V and Merge Channels
        split(hsv, hsv_channels);
        equalizeHist(hsv_channels[2], hsv_channels[2]);
        merge(hsv_channels, hsv);

        Mat frame_normalized;
        cvtColor(hsv, frame_normalized, COLOR_HSV2BGR);

        Mat output = frame.clone();
        int index = 1;
        vector<tuple<double, int, int>> seen_contours;

        // Count of Each Sticker Color
        map<string, int> cnt_colors = {
            {"green", 0}, {"white", 0}, {"yellow", 0}, {"orange", 0}, {"blue", 0}, {"red", 0}
        };

        imshow("HSV", frame_normalized);

        // Detecting Color Regions 
        for (const auto& [color, values] : color_ranges) {
            Scalar lower = get<0>(values);
            Scalar upper = get<1>(values);
            Scalar bgr = get<2>(values);

            Mat mask, edges, dilated;
            inRange(hsv, lower, upper, mask);
            Canny(mask, edges, 5, 90);
            Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
            dilate(edges, dilated, kernel, Point(-1, -1), 2);

            vector<vector<Point>> contours;
            findContours(dilated, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

            for (const auto& cnt : contours) {
                double area = contourArea(cnt);
                if (area < 2000) continue;

                vector<Point> approx;
                approxPolyDP(cnt, approx, 0.02 * arcLength(cnt, true), true); // Adjust epsilon as needed
                if (approx.size() > 8) continue; // Keep only quadrilaterals

                Moments M = moments(approx);
                if (M.m00 == 0) continue;
                int cx = static_cast<int>(M.m10 / M.m00);
                int cy = static_cast<int>(M.m01 / M.m00);

                // Check for duplicate/overlapping contours
                if (isDuplicate(cx, cy, seen_contours)) continue;
                seen_contours.emplace_back(area, cx, cy);
                
                // Draw and Label Centre with Number
                circle(output, Point(cx, cy), 3, Scalar(0, 0, 0), -1);
                putText(output, to_string(index), Point(cx - 10, cy - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
                drawContours(output, vector<vector<Point>>{approx}, -1, bgr, 2);
                index++;
                
                // Appending each sticker color count directly for now
                cnt_colors[color]++;
            }

            imshow("Rubik's Cube edge", dilated);
        }

        imshow("Rubik's Cube Tracking", output);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        frame_times.push_back(duration.count());

        if (waitKey(1) == 'q') break;
        frame_index++;
    }

    cap.release();
    destroyAllWindows();

    if (!frame_times.empty()) {
        double avg_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
        double fps = 1.0 / avg_time;
        cout << "\nAverage FPS: " << fps << endl;
    } else {
        cout << "No frames processed." << endl;
    }

    return 0;
}
