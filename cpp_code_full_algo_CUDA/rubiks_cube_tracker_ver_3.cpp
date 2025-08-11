#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <tuple>
#include <map>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "helper_funcs.hpp"
#include "cuda_process.hpp"

using namespace std;
using namespace cv;

static bool g_use_cuda = (cuda::getCudaEnabledDeviceCount() > 0);
static ProcCUDA g_cuda;

// Thread-safe queue for Mat
typedef queue<Mat> MatQueue;
class FrameQueue {
    MatQueue q;
    mutex mtx;
    condition_variable cond;
public:
    void push(const Mat& frame) {
        {
            lock_guard<mutex> lk(mtx);
            // if (!q.empty()) {
            //     // drop the old frame to keep only the latest
            //     q.pop();
            // }
            q.push(frame.clone());
        }
        cond.notify_one();
    }

    // Blocks until a frame is available
    bool pop(Mat& frame) {
        unique_lock<mutex> lk(mtx);
        cond.wait(lk, [&]{ return !q.empty(); });
        frame = move(q.front());
        q.pop();
        return !frame.empty();
    }
};


// Thread 1a: Capture camera frames using V4L2 non-blocking
void captureLoop(const string& device, FrameQueue& rawQ) {
    // Open device in non-blocking mode
    int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        cerr << "Error opening " << device << ": " << strerror(errno) << endl;
        rawQ.push(Mat());
        return;
    }

    // Set format (YUYV, 640x480)
    v4l2_format fmt = {};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = 640;
    fmt.fmt.pix.height      = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        close(fd);
        rawQ.push(Mat());
        return;
    }

    // Request and mmap buffers
    v4l2_requestbuffers req = {};
    req.count  = 2; // Ring Buffer size of 2 to hold frames
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        close(fd);
        rawQ.push(Mat());
        return;
    }

    struct Buffer { void* start; size_t length; };
    vector<Buffer> bufs(req.count);
    for (int i = 0; i < (int)req.count; ++i) {
        v4l2_buffer buf = {};
        buf.type   = req.type;
        buf.memory = req.memory;
        buf.index  = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            close(fd);
            rawQ.push(Mat());
            return;
        }
        bufs[i].length = buf.length;
        bufs[i].start  = mmap(nullptr, buf.length,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, buf.m.offset);
        
        if (bufs[i].start == MAP_FAILED) {
            perror("mmap");
            close(fd);
            rawQ.push(Mat());
            return;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            close(fd);
            rawQ.push(Mat());
            return;
        }
    }

    // Start streaming
    int type = req.type;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        close(fd);
        rawQ.push(Mat());
        return;
    }

    // Non-blocking capture loop
    v4l2_buffer buf = {};
    buf.type   = req.type;
    buf.memory = req.memory;
    while (true) {
        int ret = ioctl(fd, VIDIOC_DQBUF, &buf);
        if (ret < 0) {
            if (errno == EAGAIN) {
                this_thread::sleep_for(chrono::microseconds(100));
                continue;
            } else {
                perror("VIDIOC_DQBUF");
                break;
            }
        }

        // Wrap YUYV data into Mat and convert to BGR
        Mat yuyv(fmt.fmt.pix.height,
                 fmt.fmt.pix.width,
                 CV_8UC2,
                 bufs[buf.index].start);
        Mat bgr;
        cvtColor(yuyv, bgr, COLOR_YUV2BGR_YUYV);
        rawQ.push(bgr);

        // Re-queue buffer
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            break;
        }
    }

    // Cleanup
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (auto &b : bufs) munmap(b.start, b.length);
    close(fd);
    rawQ.push(Mat());
}

// Thread 1b: Input a Video File
void fileCaptureLoop(const string& filename, FrameQueue& rawQ) {
    VideoCapture cap(filename);
    if (!cap.isOpened()) {
        cerr << "Error opening file " << filename << endl;
        rawQ.push(Mat());
        return;
    }
    Mat frame;
    while (cap.read(frame)) {
        rawQ.push(frame);
        if (waitKey(1) == 'q') break;
    }
    rawQ.push(Mat());
}


// Thread 2: Process frames
void processingLoop(FrameQueue& rawQ, FrameQueue& procQ) {

    static int frame_index = 0;
    vector<double> frame_times;
    Mat frame;

    while (rawQ.pop(frame)) {

        auto start = chrono::high_resolution_clock::now();

        set<int> used_indices; // Keep track of all points used in any detected quad

        Mat hsv, gray, closed;

        // Run Image Transformation on GPU through CUDA
        g_cuda.run(frame, hsv, gray, closed);
        imshow("Transformed Image before Contour Detection", closed);

        // Find contours in the closed image
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Detect Stickers and Classify Colors
        Mat output = frame.clone();
        vector<tuple<double,int,int>> seen_contours;
        vector<Center> centers;
        int idx = 1;

        static Point2f dbgP1, dbgP2, dbgP3;
        for (auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < 300 || area > 10000) continue;

            // shape approximation
            vector<Point> approx;
            approxPolyDP(cnt, approx, 0.05*arcLength(cnt,true), true);
            if (approx.size() > 8) continue;

            // mask & mean HSV
            Mat mask = Mat::zeros(frame.size(), CV_8U);
            drawContours(mask, vector<vector<Point>>{approx}, -1, 255, FILLED);
            
            Scalar m = mean(hsv, mask); // Returns H, S, V, A
            Vec3f hsvAvg((float)m[0], (float)m[1], (float)m[2]);

            // color classification
            string cname = classifyColor(hsvAvg);

            // centroid
            Moments Mo = moments(approx);
            if (Mo.m00 == 0) continue;
            int cx = int(Mo.m10/Mo.m00), cy = int(Mo.m01/Mo.m00);
            if (isDuplicate(cx, cy, seen_contours)) continue;
            seen_contours.emplace_back(area, cx, cy);
            // centers.push_back(Center{idx, cx, cy, hsvAvg});
            centers.push_back({idx, cx, cy, cname});

            // drawColor lookup
            Scalar drawColor(0,0,0);
            auto it = color_ranges.find(cname);
            if (it != color_ranges.end())
                drawColor = get<2>(it->second);

            // Draw Center, Text and Colored Contour
            // circle(output, Point(cx,cy), 3, Scalar(0,0,0), FILLED);
            // putText(output, to_string(idx), Point(cx-10,cy-10),
            //         FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 2);
            // drawContours(output, vector<vector<Point>>{approx}, -1, drawColor, 2);

            idx++;
        }

        vector<Face> faces; // Initialize before the loop that finds each face

        // Check for every Triplet to build and validate Quad 
        int n = (int)centers.size();
        for (int p = 1; p <= n; ++p) {
            if (used_indices.count(p)) continue;
            Center* p1 = findByIndex(centers, p);
            if (!p1) continue;
            for (int q = p+1; q <= n; ++q) {
                if (used_indices.count(q)) continue;    
                Center* p2 = findByIndex(centers, q);
                if (!p2) continue;
                for (int r = q+1; r <= n; ++r) {
                    if (used_indices.count(r)) continue;
                    Center* p3 = findByIndex(centers, r);
                    if (!p3) continue;

                    // 1) Collinearity test via triangle area
                    double A = triangleArea(
                        p1->cx, p1->cy,
                        p2->cx, p2->cy,
                        p3->cx, p3->cy
                    );
                    if (A >= AREA_TOLERANCE) 
                        continue;

                    // 2) Identify the longest of the three sides → (start,end,middle)
                    struct Trip { double d; Center* a; Center* b; Center* c; };
                    vector<Trip> trips = {
                        { dist(*p1, *p2), p1, p2, p3 },
                        { dist(*p2, *p3), p2, p3, p1 },
                        { dist(*p1, *p3), p1, p3, p2 },
                    };
                    auto best = *max_element(trips.begin(), trips.end(),
                        [](auto &L, auto &R){ return L.d < R.d; }
                    );
                    Center* start  = best.a;
                    Center* end    = best.b;
                    Center* middle = best.c;

                    // 3) Try to build a quad from that base edge
                    auto firstLine = make_tuple(start, end, middle);
                    auto quadLines = buildQuadFromStartLine(
                        centers,
                        firstLine,
                        AREA_TOLERANCE
                    );
                    if (quadLines.empty())
                        continue;

                    // 4) Extract the four corner points
                    vector<Center*> quadPts;
                    quadPts.reserve(4); // CHANGED FROM 5 to 4
                    for (auto &ln : quadLines)
                        quadPts.push_back(get<0>(ln));

                    // 5) Validate and draw
                    if (quadPts.size() == 4 && isParallelogram(quadPts) && isMiddleBalanced(quadLines)) {
                        drawQuadOnFrame(
                            output,
                            quadPts[0], quadPts[1],
                            quadPts[2], quadPts[3],
                            Scalar(255,0,255), 2
                        );
                        // cout << "Frame " << frame_index
                            // << ": parallelogram passed!" << endl;

                        set<int> quad_indices;
                        for (const auto& ln : quadLines) {
                            quad_indices.insert(get<0>(ln)->index);  // start
                            quad_indices.insert(get<1>(ln)->index);  // end
                            quad_indices.insert(get<2>(ln)->index);  // middle
                        }

                            // Add them to the global used_indices set
                        used_indices.insert(quad_indices.begin(), quad_indices.end());

                        // Draw each point in the quad with its detected color
                        for (int idx_used : quad_indices) {
                            const Center* pt = findByIndex(centers, idx_used);
                            if (!pt) continue;

                            Scalar bgr_color(0,0,0); 
                            auto it = color_ranges.find(pt->hsv_color_name);
                            if (it != color_ranges.end())
                                bgr_color = get<2>(it->second);       // BGR draw color

                            circle(output, Point(pt->cx, pt->cy), 8, bgr_color, FILLED);
                        }
                        
                        // Get polygon vertices as (x, y) tuples
                        vector<Point> poly_pts;
                        poly_pts.reserve(4);
                        for (int i = 0; i < 4 && i < static_cast<int>(quadPts.size()); ++i)
                            poly_pts.emplace_back(quadPts[i]->cx, quadPts[i]->cy);

                        // Store the centers that belong to the face
                        vector<Center*> inside_centers;
                        vector<int> inside_indices;

                        // Find centers inside the parallelogram
                        for (auto &c : centers) {
                            double res = pointPolygonTest(poly_pts, Point2f(static_cast<float>(c.cx),
                                                                                    static_cast<float>(c.cy)), false);
                            if (res >= 0) {  // inside or on edge
                                inside_indices.push_back(c.index);
                                inside_centers.push_back(&c);
                            }
                        }

                        if (!inside_centers.empty()) {
                            long sumx = 0, sumy = 0;
                            for (const Center* c : inside_centers) {
                                sumx += c->cx;
                                sumy += c->cy;
                            }
                            int mean_x = static_cast<int>(sumx / static_cast<long>(inside_centers.size()));
                            int mean_y = static_cast<int>(sumy / static_cast<long>(inside_centers.size()));
                            faces.push_back(Face{Point(mean_x, mean_y), inside_indices });
                        }
                        
                        // Get the centers for all stickers belonging to the face from quad_indices
                        vector<Center*> face_stickers;
                        face_stickers.reserve(quad_indices.size());
                        for (int idx_used : quad_indices) {
                            Center* p = findByIndex(centers, idx_used);
                            if (p) face_stickers.push_back(p);
                        }

                        // Order corners and assign positions by parallelogram
                        vector<Center*> ordered_corners = orderParallelogramCorners(quadPts); // TL, TR, BL, BR
                        auto positions = assignPositionsByParallelogram(face_stickers, ordered_corners);
                        
                        
                        // For each sticker position (index -> (row,col))
                        // for (const auto& kv : positions) {
                        //     int idx = kv.first;
                        //     int row = kv.second.first;
                        //     int col = kv.second.second;

                        // After all three faces have been collected:

                        if (faces.size() == 3) {
                            // Step 1: Top face = smallest y
                            auto it_top = min_element(faces.begin(), faces.end(),
                                [](const Face& a, const Face& b){ return a.center.y < b.center.y; });
                            const Face* top_face = &(*it_top);

                            // Step 2: Right face = largest x among remaining
                            vector<const Face*> remaining;
                            remaining.reserve(2);
                            for (const auto& f : faces)
                                if (&f != top_face) remaining.push_back(&f);

                            // Step 3: right = largest x among remaining
                            auto it_right = max_element(
                                remaining.begin(), remaining.end(),
                                [](const Face* a, const Face* b){ return a->center.x < b->center.x; });
                            const Face* right_face = *it_right;

                            // 4) Front face = the other one
                            const Face* front_face = (remaining[0] == right_face) ? remaining[1] : remaining[0];

                            // Print each sticker on the FRONT face
                            for (int idx : front_face->indices) {
                                Center* sticker = findByIndex(centers, idx);
                                if (!sticker) continue;

                                int cx = sticker->cx, cy = sticker->cy;
                                
                                // You store the classified color name in hsv_color_name
                                cout << "Index " << idx
                                        << ": COLOR=" << sticker->hsv_color_name
                                        << " -> (" << cx << ", " << cy << ")\n";
                            }

                            // Helper to print index lists
                            auto print_indices = [](const char* label, const std::vector<int>& v){
                                std::cout << label << ": ";
                                for (int i : v) std::cout << i << ' ';
                                std::cout << '\n';
                            };

                            print_indices("TOP",   top_face->indices);
                            print_indices("FRONT", front_face->indices);
                            print_indices("RIGHT", right_face->indices);

                            
                            cv::putText(output, "TOP",   top_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                            cv::putText(output, "FRONT", front_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                            cv::putText(output, "RIGHT", right_face->center,
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
                        }

                        else{
                            cout << "Only " << faces.size() << " faces detected this frame\n";
                        }

                    }    
                                            
                }
            }
        }

        // Push Processed Frame and Record Time
        procQ.push(output);
        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();
        frame_times.push_back(elapsed);
        ++frame_index;
    }

    // Signal end of processing
    procQ.push(Mat());

    // Print Average FPS
    if (!frame_times.empty()) {
        double avg_time = accumulate(frame_times.begin(), frame_times.end(), 0.0)
                      / frame_times.size();
        double fps = 1.0 / avg_time;
        cout << "\nAverage FPS: " << fps << endl;
    } else {
        cout << "No frames processed." << endl;
    }
}

// Thread 3: Display results
void displayLoop(FrameQueue& procQ) {
    Mat out;
    while (procQ.pop(out)) {
        imshow("Rubik's Cube Tracking", out);
        if (waitKey(1) == 'q') break;
    }
}


int main(int argc, char** argv) {
    if (argc != 3 || (string(argv[1])!="-f" && string(argv[1])!="-d")) {
        cerr << "Usage: " << argv[0] << " -f <video_file> | -d <device>\n";
        return -1;
    }
    
    string mode = argv[1], source = argv[2];
    FrameQueue rawQ, procQ;
    thread tCap;
    
    if (mode == "-f") 
        tCap = thread(fileCaptureLoop, source, ref(rawQ));
    else              
        tCap = thread(captureLoop, source, ref(rawQ));

    thread tProc(processingLoop, ref(rawQ), ref(procQ));
    thread tDisp(displayLoop, ref(procQ));
    tCap.join(); tProc.join(); tDisp.join();

    return 0;
}
