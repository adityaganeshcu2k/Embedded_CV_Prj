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

using namespace std;
using namespace cv;

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

// Check for Duplicate Contours
bool isDuplicate(int cx, int cy, const vector<tuple<double,int,int>>& seen, int thresh = 10) {
    for (auto& t : seen) {
        int sx = get<1>(t), sy = get<2>(t);
        if (abs(cx - sx) < thresh && abs(cy - sy) < thresh)
            return true;
    }
    return false;
}

// Thread 1: Capture raw frames using V4L2 non-blocking
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

// Thread 2: Process frames
void processingLoop(FrameQueue& rawQ, FrameQueue& procQ) {
    map<string, tuple<Scalar,Scalar,Scalar>> color_ranges = {
        {"white", {Scalar(0,0,175), Scalar(180,40,255), Scalar(255,255,255)}}
    };

    Mat frame;
    while (rawQ.pop(frame)) {
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        vector<Mat> ch;
        split(hsv, ch);
        equalizeHist(ch[2], ch[2]);
        merge(ch, hsv);

        Mat output = frame.clone();
        int idx = 1;
        vector<tuple<double,int,int>> seen;

        for (auto& kv : color_ranges) {
            const auto& [low, high, drawColor] = kv.second;
            Mat mask, edges, dil;
            inRange(hsv, low, high, mask);
            Canny(mask, edges, 5, 90);
            dilate(edges, dil, getStructuringElement(MORPH_RECT, {2,2}), Point(-1,-1), 2);

            vector<vector<Point>> contours;
            findContours(dil, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

            for (auto& cnt : contours) {
                double area = contourArea(cnt);
                if (area < 2000) continue;
                vector<Point> approx;
                approxPolyDP(cnt, approx, 0.02 * arcLength(cnt, true), true);
                if (approx.size() > 8) continue;
                Moments M = moments(approx);
                if (M.m00 == 0) continue;
                int cx = int(M.m10 / M.m00), cy = int(M.m01 / M.m00);
                if (isDuplicate(cx, cy, seen)) continue;
                seen.emplace_back(area, cx, cy);

                circle(output, Point(cx,cy), 3, Scalar(0,0,0), -1);
                putText(output, to_string(idx++), Point(cx-10,cy-10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 2);
                drawContours(output, vector<vector<Point>>{approx}, -1, drawColor, 2);
            }
        }

        procQ.push(output);
    }
    procQ.push(Mat());
}

// Thread 3: Display results
void displayLoop(FrameQueue& procQ) {
    Mat out;
    while (procQ.pop(out)) {
        imshow("Rubik's Cube Tracking", out);
        if (waitKey(1) == 'q') break;
    }
}

int main() {
    FrameQueue rawQ, procQ;

    thread tCap(captureLoop, "/dev/video0", ref(rawQ));
    thread tProc(processingLoop, ref(rawQ), ref(procQ));
    thread tDisp(displayLoop, ref(procQ));

    tCap.join();
    tProc.join();
    tDisp.join();

    return 0;
}
