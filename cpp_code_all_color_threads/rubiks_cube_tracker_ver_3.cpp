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

using namespace std;
using namespace cv;

constexpr double AREA_TOLERANCE = 700.00;

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
            if (!q.empty()) {
                // drop the old frame to keep only the latest
                q.pop();
            }
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

// Map for All Color Ranges
map<string, tuple<Scalar,Scalar,Scalar>> color_ranges = {
    {"white",  { Scalar(  0,   0, 200), Scalar(180,  50, 255), Scalar(255, 255, 255) }},  // low saturation, high value
    {"yellow", { Scalar( 20, 100, 100), Scalar( 35, 255, 255), Scalar(  0, 255, 255) }},
    {"orange", { Scalar( 10, 100, 100), Scalar( 20, 255, 255), Scalar(  0, 165, 255) }},
    {"blue",   { Scalar( 90, 100, 100), Scalar(130, 255, 255), Scalar(255,   0,   0) }},
    {"green",  { Scalar( 40,  70, 100), Scalar( 85, 255, 255), Scalar(  0, 255,   0) }},
    {"red1",   { Scalar(  0, 100, 100), Scalar( 10, 255, 255), Scalar(  0,   0, 255) }},  // red lower hue
    {"red2",   { Scalar(160, 100, 100), Scalar(180, 255, 255), Scalar(  0,   0, 255) }}   // red upper hue
};

// Compute the midpoint of two HSV Scalar bounds
static Vec3f getAverageColor(const Scalar& lower, const Scalar& upper) {
    return Vec3f(
        (lower[0] + upper[0]) * 0.5f,
        (lower[1] + upper[1]) * 0.5f,
        (lower[2] + upper[2]) * 0.5f
    );
}

// Classify an HSV mean into the closest color in color_ranges
static string classifyColor(const Vec3f& hsvAvg) {
    double minDist = numeric_limits<double>::infinity();
    string best = "unknown";

    for (const auto& kv : color_ranges) {
        const string& name    = kv.first;
        const Scalar& lower   = get<0>(kv.second);
        const Scalar& upper   = get<1>(kv.second);
        // ignore the BGR draw-color at index 2 for classification

        Vec3f reference = getAverageColor(lower, upper);
        // Euclidean distance in HSV-space
        double d = norm(hsvAvg - reference);

        if (d < minDist) {
            minDist = d;
            best    = name;
        }
    }

    return best;
}

// Check for Duplicate Contours
bool isDuplicate(int cx, int cy, const vector<tuple<double,int,int>>& seen, int thresh = 10) {
    for (auto& t : seen) {
        int sx = get<1>(t), sy = get<2>(t);
        if (abs(cx - sx) < thresh && abs(cy - sy) < thresh)
            return true;
    }
    return false;
}

// Defining Center structure and associated functions
struct Center {
    int index;
    int cx, cy;
};

// Find by Index Function
Center* findByIndex(vector<Center>& centers, int idx) {
    for (auto& c : centers) {
        if (c.index == idx) return &c;
    }
    return nullptr;
}

// Triangle Area Function
double triangleArea(int x1, int y1, int x2, int y2, int x3, int y3) {
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
}

// Distance between Centers Function
double dist(const Center& p1, const Center& p2) {
    return hypot(p2.cx - p1.cx, p2.cy - p1.cy);
}

// Midpoint Function
Point midpoint(const Center& p1, const Center& p2) {
    return Point((p1.cx + p2.cx) / 2, (p1.cy + p2.cy) / 2);
}

// Angle Between Vectors
double angleBetween(const Point2d& v1, const Point2d& v2) {
    double dot = v1.dot(v2);
    double mag1 = norm(v1);
    double mag2 = norm(v2);
    if (mag1 == 0 || mag2 == 0) return 0;
    double cosTheta = clamp(dot / (mag1 * mag2), -1.0, 1.0);
    return acos(cosTheta) * 180.0 / CV_PI;
}

// Find next Collinear Side Function
// Returns all (start, end, middle) triples of points nearly collinear with fixedPoint
static vector<tuple<Center*, Center*, Center*>> findAllNextCollinearSides(
    vector<Center>& centers,
    Center* fixedPoint,
    double areaTol=AREA_TOLERANCE,
    const set<int>& usedPoints = {},
    Center* allowPoint = nullptr)
{
    // 1) build candidate list
    vector<Center*> candidates;
    for (auto& c : centers) {
        if (&c == fixedPoint) continue;
        if (usedPoints.count(c.index)) {
            if (allowPoint && allowPoint->index == c.index)
                candidates.push_back(&c);
        } else {
            candidates.push_back(&c);
        }
    }

    vector<tuple<Center*, Center*, Center*>> results;
    // 2) every pair (p, q) among candidates
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            Center* p = candidates[i];
            Center* q = candidates[j];

            // 3) collinearity via small triangle area
            double A = triangleArea(
                fixedPoint->cx, fixedPoint->cy,
                p->cx, p->cy,
                q->cx, q->cy
            );
            if (A >= areaTol) continue;

            // 4) compute the three segment lengths
            double d1 = dist(*fixedPoint, *p);
            double d2 = dist(*p, *q);
            double d3 = dist(*fixedPoint, *q);

            struct Trip { double d; Center* a; Center* b; Center* c; };
            vector<Trip> trips = {
                {d1, fixedPoint, p, q},
                {d2, p, q, fixedPoint},
                {d3, fixedPoint, q, p}
            };

            // 5) pick the longest segment → (start, end, middle)
            auto best = max_element(
                trips.begin(), trips.end(),
                [](auto &L, auto &R){ return L.d < R.d; }
            );
            Center* start  = best->a;
            Center* end    = best->b;
            Center* middle = best->c;

            // 6) ensure fixedPoint is one endpoint
            if (fixedPoint != start && fixedPoint != end)
                continue;
            if (start != fixedPoint)
                std::swap(start, end);

            results.emplace_back(start, end, middle);
        }
    }

    return results;
}


// Build Quad from Start Line Function
// Returns an empty vector on failure, or the list of 4 collinear‐side triples on success.
static vector<tuple<Center*,Center*,Center*>> buildQuadFromStartLine(
    vector<Center>& centers,
    const tuple<Center*,Center*,Center*>& firstLine,
    double areaTol = AREA_TOLERANCE,
    double sideRatioThresh = 1.5,
    const pair<double,double>& angleThresh = pair<double,double>(30.0, 150.0),
    set<int> usedPoints = {},
    vector<tuple<Center*,Center*,Center*>> lines = {},
    int depth = 0)
{
    if (depth == 0) {
        usedPoints = {
            get<0>(firstLine)->index,
            get<1>(firstLine)->index,
            get<2>(firstLine)->index
        };
        lines = { firstLine };
    }

    // Early angle check when depth>=2
    if (depth >= 2) {
        int sz = (int)lines.size();
        auto &tpl_prev = lines[sz-2];
        Center *p_prev = get<0>(tpl_prev), *p_curr = get<1>(tpl_prev);
        auto &tpl_curr = lines[sz-1];
        Center *p_next = get<1>(tpl_curr);

        Point2d v1(p_prev->cx - p_curr->cx, p_prev->cy - p_curr->cy);
        Point2d v2(p_next->cx - p_curr->cx, p_next->cy - p_curr->cy);
        double ang = angleBetween(v1, v2);
        if (ang < angleThresh.first || ang > angleThresh.second)
            return {};
    }

    // Completion check at depth==3
    if (depth == 3) {
        auto &last_tpl  = lines.back();
        Center *last_end   = get<1>(last_tpl);
        auto &first_tpl = lines.front();
        Center *first_start = get<0>(first_tpl);
        if (last_end->index != first_start->index) return {};

        // Side‐length ratio
        vector<double> sides;
        sides.reserve(4);
        for (auto &ln : lines) {
            Center *a = get<0>(ln), *b = get<1>(ln);
            sides.push_back(dist(*a, *b));
        }
        auto [minIt, maxIt] = minmax_element(sides.begin(), sides.end());
        if (*maxIt > sideRatioThresh * *minIt) return {};

        // Build unique corner list
        vector<Center*> quad;
        quad.reserve(4);
        for (auto &ln : lines) {
            Center *a = get<0>(ln);
            if (quad.empty() || quad.back()->index != a->index)
                quad.push_back(a);
        }
        if (get<1>(lines.back())->index != quad.front()->index)
            quad.push_back(get<1>(lines.back()));
        if ((int)quad.size() != 4) return {};

        // Final corner‐angle test
        for (int i = 0; i < 4; ++i) {
            Center *prev = quad[(i+3)%4], *curr = quad[i], *next = quad[(i+1)%4];
            Point2d va(prev->cx - curr->cx, prev->cy - curr->cy);
            Point2d vb(next->cx - curr->cx, next->cy - curr->cy);
            double a = angleBetween(va, vb);
            if (a < angleThresh.first || a > angleThresh.second)
                return {};
        }
        // success!
        return lines;
    }

    // Extend the quad by finding collinear sides off the last endpoint
    Center* fixedPoint = get<1>(lines.back());
    auto nextLines = findAllNextCollinearSides(
        centers,
        fixedPoint,
        areaTol,
        usedPoints,
        depth == 2 ? get<0>(firstLine) : nullptr
    );

    for (auto &nl : nextLines) {
        Center *endPt = get<1>(nl);
        if (usedPoints.count(endPt->index) && endPt != get<0>(firstLine))
            continue;

        auto newUsed  = usedPoints;
        newUsed.insert(endPt->index);

        auto newLines = lines;
        newLines.push_back(nl);  // nl is tuple<Center*,Center*,Center*>

        auto result = buildQuadFromStartLine(
            centers,
            firstLine,
            areaTol,
            sideRatioThresh,
            angleThresh,
            newUsed,
            newLines,
            depth + 1
        );
        if (!result.empty())
            return result;
    }

    return {};
}


// Is Parallelogram Function
static bool isParallelogram(
    const vector<Center*>& pts,
    double lengthRatioTol = 0.25,
    double angleTolDeg    = 10.0)
{
    if (pts.size() != 4)
        throw std::invalid_argument("Exactly 4 points required");

    Center *A = pts[0], *B = pts[1], *C = pts[2], *D = pts[3];

    // build the four side vectors
    Point2d AB(B->cx - A->cx, B->cy - A->cy);
    Point2d BC(C->cx - B->cx, C->cy - B->cy);
    Point2d CD(D->cx - C->cx, D->cy - C->cy);
    Point2d DA(A->cx - D->cx, A->cy - D->cy);

    // lengths
    double lenAB = norm(AB), lenBC = norm(BC);
    double lenCD = norm(CD), lenDA = norm(DA);

    // side‐length ratio checks
    double r1 = lenCD / (lenAB + 1e-10);
    if (r1 < 1 - lengthRatioTol || r1 > 1 + lengthRatioTol)
        return false;

    double r2 = lenDA / (lenBC + 1e-10);
    if (r2 < 1 - lengthRatioTol || r2 > 1 + lengthRatioTol)
        return false;

    // angle between opposite sides
    double angAB_CD = angleBetween(AB, CD);
    if (!(angAB_CD < angleTolDeg || std::abs(angAB_CD - 180.0) < angleTolDeg))
        return false;

    double angBC_DA = angleBetween(BC, DA);
    if (!(angBC_DA < angleTolDeg || std::abs(angBC_DA - 180.0) < angleTolDeg))
        return false;

    return true;
}


// Middle-balance Function
static bool isMiddleBalanced(
    const vector<tuple<Center*,Center*,Center*>>& lines,
    double ratioThresh = 1.5)
{
    for (size_t i = 0; i < lines.size(); ++i) {
        Center* start  = get<0>(lines[i]);
        Center* end    = get<1>(lines[i]);
        Center* middle = get<2>(lines[i]);

        double d1 = dist(*start, *middle);
        double d2 = dist(*end,   *middle);

        double shorter = min(d1, d2);
        double longer  = max(d1, d2);

        if (shorter < 1e-3) {
            cout << "Line " << (i+1) << ": Too short to compare reliably." << endl;
            return false;
        }

        double ratio = longer / shorter;
        if (ratio > ratioThresh) {
            cout << "Line " << (i+1)
                 << " failed middle-balance check: ratio = "
                 << ratio << " (limit " << ratioThresh << ")" << endl;
            return false;
        }
    }
    return true;
}

// Draw Quad on Frame Function, connecting 4 centers on frame
static void drawQuadOnFrame(
    Mat& frame,
    Center* p1, Center* p2, Center* p3, Center* p4,
    const Scalar& color = Scalar(255, 0, 255),
    int thickness = 2)
{
    // 1) Gather the four corner points
    vector<Point> pts = {
        Point(p1->cx, p1->cy),
        Point(p2->cx, p2->cy),
        Point(p3->cx, p3->cy),
        Point(p4->cx, p4->cy)
    };

    // 2) Compute centroid as float sum
    Point2f center(0.f, 0.f);
    for (const auto& pt : pts) {
        center.x += static_cast<float>(pt.x);
        center.y += static_cast<float>(pt.y);
    }
    center.x /= static_cast<float>(pts.size());
    center.y /= static_cast<float>(pts.size());

    // 3) Sort points by angle around centroid (clockwise)
    sort(pts.begin(), pts.end(),
        [&center](const Point& a, const Point& b) {
            double angA = atan2(a.y - center.y, a.x - center.x);
            double angB = atan2(b.y - center.y, b.x - center.x);
            return angA < angB;
        }
    );

    // 4) Draw the closed quadrilateral
    vector<vector<Point>> contour = { pts };
    polylines(frame, contour, true, color, thickness);
}


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

        Mat hsv, gray, thresh;
        
        // Convert to HSV
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Convert to Greyscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Adaptive thresholding (Gaussian)
        adaptiveThreshold(
            gray,                 // source
            thresh,               // destination
            255,                  // maxValue
            ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY_INV,
            21,                   // blockSize
            4                     // C
        );

        // Create a 2×2 structuring element
        Mat kernel = Mat::ones(Size(2, 2), CV_8U);

        // Dilate to thicken edges (5 iterations)
        Mat dilated_edges;
        dilate(thresh, dilated_edges, kernel, Point(-1, -1), 5);

        // Close small gaps inside sticker areas (2 iterations)
        Mat closed;
        morphologyEx(dilated_edges, closed, MORPH_CLOSE, kernel, Point(-1, -1), 2);

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
            if (area < 1000 || area > 10000) continue;

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
            centers.push_back({idx, cx, cy});

            // drawColor lookup
            Scalar drawColor(0,0,0);
            auto it = color_ranges.find(cname);
            if (it != color_ranges.end())
                drawColor = get<2>(it->second);

            // debug frame 463
            // if (frame_index == 463) {
            //     if (idx == 13) dbgP1 = Point2f(cx, cy);
            //     else if (idx == 21) dbgP2 = Point2f(cx, cy);
            //     else if (idx == 4)  dbgP3 = Point2f(cx, cy);
            // }

            // Draw Cebterm Text and Colored Contour
            circle(output, Point(cx,cy), 3, Scalar(0,0,0), FILLED);
            putText(output, to_string(idx), Point(cx-10,cy-10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 2);
            drawContours(output, vector<vector<Point>>{approx}, -1, drawColor, 2);

            idx++;
        }

        // Check for every Triplet to build and validate Quad 
        int n = (int)centers.size();
        for (int p = 1; p <= n; ++p) {
            Center* p1 = findByIndex(centers, p);
            if (!p1) continue;
            for (int q = p+1; q <= n; ++q) {
                Center* p2 = findByIndex(centers, q);
                if (!p2) continue;
                for (int r = q+1; r <= n; ++r) {
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
                        cout << "Frame " << frame_index
                            << ": parallelogram passed!" << endl;
                    }
                }
            }
        }

        // Push Processed Frame and Record Time
        procQ.push(output);
        auto end = chrono::high_resolution_clock::now();
        // elapsed seconds for this frame
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
